import torch
import torch.nn as nn
from eg_mamba.modules.utils.processing_timer import ProcessingTimer


class AggregationFusion(nn.Module):
    def __init__(self, fusion_type="sum", node_dim=None):
        """
        Args:
            fusion_type: "sum", "concat", "gate", "mlp_res".
            node_dim: If fusion_type is not "sum", the dimension of the original node features is usually required.
        """
        super().__init__()
        self.fusion_type = fusion_type.lower()
        if self.fusion_type not in {"sum", "concat", "gate", "mlp_res", "sigmoid"}:
            raise ValueError("fusion_type must be one of 'sum', 'concat', 'gate', or 'mlp_res'.")

        if self.fusion_type in {"concat", "gate"} and node_dim is None:
            raise ValueError("When using concat or gate fusion methods, the node_dim parameter must be specified.")

        self.node_dim = node_dim
        if self.fusion_type == "concat":
            self.concat_layer = nn.Sequential(
                # nn.RMSNorm(node_dim * 2),
                nn.LayerNorm(node_dim * 2),
                nn.Linear(node_dim * 2, node_dim * 2),
                nn.SiLU(),
                nn.Dropout(0.2),

                # nn.RMSNorm(node_dim * 2),
                nn.LayerNorm(node_dim * 2),
                nn.Linear(node_dim * 2, node_dim),
                nn.SiLU(),
                nn.Dropout(0.2),
            )
        elif self.fusion_type == "gate":
            # self.aggr_proj_layer = nn.Sequential(
            #     # nn.LayerNorm(node_dim),
            #     nn.Linear(node_dim, node_dim),
            #     # nn.SiLU(),
            #     # nn.Dropout(0.2),
            # )
            self.gate_layer = nn.Sequential(
                # nn.RMSNorm(node_dim * 2),
                nn.LayerNorm(node_dim * 2),
                nn.Linear(node_dim*2, node_dim//4),
                nn.SiLU(),
                nn.Dropout(0.2),

                # nn.RMSNorm(node_dim//4),
                nn.LayerNorm(node_dim//4),
                nn.Linear(node_dim//4, 2),
                nn.Softmax(dim=-1)
            )
        elif self.fusion_type == "sigmoid":
            self.sigmoid_layer = nn.Sequential(
                # nn.RMSNorm(node_dim*2),
                nn.LayerNorm(node_dim*2),
                nn.Linear(node_dim*2, node_dim*4),
                nn.SiLU(),
                nn.Dropout(0.2),

                # nn.RMSNorm(node_dim*4),
                nn.LayerNorm(node_dim*4),
                nn.Linear(node_dim*4, node_dim*2),
                nn.Sigmoid(),
            )
        elif self.fusion_type == "mlp_res":
            self.mlp_layer = nn.Sequential(
                # nn.RMSNorm(node_dim),
                nn.LayerNorm(node_dim),
                nn.Linear(node_dim, node_dim*2),
                nn.SiLU(),
                nn.Dropout(0.2),

                # nn.RMSNorm(node_dim*2),
                nn.LayerNorm(node_dim*2),
                nn.Linear(node_dim*2, node_dim),
                nn.SiLU(),
                nn.Dropout(0.2),
            )

        self.timer = ProcessingTimer(mode="gpu")

    def forward(self, coords, nodes, comps, aggr_coords, aggr_nodes, aggr_comps):
        """
        Args:
            nodes: [N, F] 
            comps: [N] 
            aggr_nodes: [M, F] 
            aggr_comps: [M] 
        Returns:
            fused_nodes: [N, F] 
        """
        mapped_indices = torch.searchsorted(aggr_comps, comps)

        selected_aggr = aggr_nodes[mapped_indices]

        self.timer.start()
        if self.fusion_type == "sum":
            fused_nodes = nodes + selected_aggr
        
        elif self.fusion_type == "concat":
            combined = torch.cat([nodes, selected_aggr], dim=-1)
            fused_nodes = self.concat_layer(combined)
        
        elif self.fusion_type == "gate":
            combined = torch.cat([nodes, selected_aggr], dim=-1)
            gating = self.gate_layer(combined)  # [N,2]
            weight_orig = gating[:, 0].unsqueeze(-1)
            weight_aggr = gating[:, 1].unsqueeze(-1)
            fused_nodes = nodes * weight_orig + selected_aggr * weight_aggr
        
        elif self.fusion_type == "mlp_res":
            transformed_aggr = self.mlp_layer(selected_aggr)
            fused_nodes = nodes + transformed_aggr
        
        elif self.fusion_type == "sigmoid":
            combined = torch.cat([nodes, selected_aggr], dim=-1)
            sigmoid_orig_aggr_weights = self.sigmoid_layer(combined)  # [N, 2*F]
            sigmoid_orig_weights = sigmoid_orig_aggr_weights[:, 0:self.node_dim] # [N, F]
            sigmoid_aggr_weights = sigmoid_orig_aggr_weights[:, self.node_dim:self.node_dim*2] # [N, F]
            fused_nodes = nodes * sigmoid_orig_weights + selected_aggr * sigmoid_aggr_weights

        else:
            raise ValueError("Unsupported fusion_type")
        
        _, gpu_time = self.timer.stop()

        return coords, fused_nodes, comps, gpu_time

