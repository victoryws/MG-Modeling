import torch
import torch.nn as nn
from eg_mamba.modules.flexible_gnn import FlexibleGNN
from eg_mamba.modules.graph_utils import get_activation, get_normalization


class FlexibleUniqueCoordinatesGenerator(nn.Module):
    def __init__(self, 
                 dim, 
                #  perturb_dim=3, 
                 drop_rate=0.2,
                 if_update_coords=True,
                 ):
        super().__init__()
        self.if_update_coords = if_update_coords
        # self.perturb_dim = perturb_dim
        
        if self.if_update_coords:
            # --- 1. Local ---
            self.local_in_proj = nn.Sequential(
                # nn.RMSNorm(dim),
                nn.LayerNorm(dim),
                nn.Linear(dim, 2*dim),
            )
            self.local_conv = nn.Conv1d(
                in_channels=2*dim,
                out_channels=2*dim,
                kernel_size=3,
                padding=1,
                groups=2*dim,
            )
            self.local_silu = nn.SiLU()
            self.local_dropout = nn.Dropout(drop_rate)
            self.local_out_proj = nn.Sequential(
                # nn.RMSNorm(2*dim),
                nn.LayerNorm(2*dim),
                nn.Linear(2*dim, dim),
                nn.SiLU(),
                nn.Dropout(drop_rate),
            )

            # --- 2. Global---
            self.global_mlp = nn.Sequential(
                # nn.RMSNorm(dim),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim*2),
                nn.SiLU(),
                nn.Dropout(drop_rate),

                # nn.RMSNorm(dim*2),
                nn.LayerNorm(dim*2),
                nn.Linear(dim*2, dim),
                nn.SiLU(),
                nn.Dropout(drop_rate),
            )
            
            # --- 3. Fusion ---
            fusion_hidden_dim = max(dim // 4, 4)
            self.fusion_head = nn.Sequential(
                # nn.RMSNorm(dim),
                nn.LayerNorm(dim),
                nn.Linear(dim, fusion_hidden_dim),
                nn.SiLU(),
                nn.Dropout(drop_rate),

                # nn.RMSNorm(fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.Linear(fusion_hidden_dim, 3),
                nn.Tanh()
            )

            self.perturb_scale_xyt = nn.Parameter(torch.tensor([0.0001, 0.0001, 0.0001]), requires_grad=True)
            self.reset_parameters()
        else:
            pass

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.local_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Global MLP
        self.global_mlp.apply(init_weights)
        
        # Fusion Head
        self.fusion_head.apply(init_weights)
            
    def forward(self, graph):
        if self.if_update_coords:
            node_features = graph.x         # [N, D]
            orig_coords = graph.orig_coords # [N, 3]
            
            # --- 1. Local ---
            local_in = self.local_in_proj(node_features)  # [N, 2D]
            # [N, 2D] -> [1, 2D, N]
            local_in_conv = local_in.unsqueeze(0).transpose(1, 2)
            local_conv_out = self.local_conv(local_in_conv)
            # [1, 2D, N] -> [N, 2D]
            local_out = self.local_silu(local_conv_out.transpose(1, 2).squeeze(0))
            local_out = self.local_dropout(local_out) # [N, 2D]
            # [N, 2D] -> [N, D]
            local_out = self.local_out_proj(local_out) # [N, D]
            

            # --- 2. Global ---
            # self.global_mlp: Norm -> Linear -> SiLU -> Linear
            global_out = self.global_mlp(node_features) # [N, D]
            
            
            # --- 3. Fusion ---
            combined = local_out + global_out # [N, D]
            delta_m = self.fusion_head(combined) # [N, perturb_dim]
            adjustment = delta_m * self.perturb_scale_xyt
            
            unique_coords = orig_coords + adjustment
        
        else:
            unique_coords = graph.orig_coords
        
        return unique_coords


def uniqueness_loss(unique_coords, threshold=1e-3):
    B_N, D = unique_coords.shape
    if B_N < 2:
        return torch.tensor(0.0, device=unique_coords.device)
    
    distance_matrix = torch.cdist(unique_coords, unique_coords, p=2)  # [B*N, B*N]
    
    mask = ~torch.eye(B_N, device=unique_coords.device).bool()
    distances = distance_matrix.masked_select(mask)  # [B*N*(B*N-1)]
    
    penalized = torch.relu(threshold - distances)  # [B*N*(B*N-1)]
    
    loss = penalized.sum() / B_N
    
    return loss

