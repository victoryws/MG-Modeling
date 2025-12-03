import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Batch
import warnings


import math
from typing import List

def get_power_of_2_hidden_dims(input_dim: int, output_dim: int, num_layers: int) -> List[int]:
    num_hidden_layers = num_layers - 1
    
    if num_hidden_layers <= 0:
        return []

    start_log = math.log2(max(input_dim, 1))
    end_log = math.log2(max(output_dim, 1))
    
    step = (end_log - start_log) / (num_hidden_layers + 1)
    
    hidden_dims = []
    for i in range(1, num_hidden_layers + 1):
        current_log = start_log + step * i
        dim_val = 2 ** round(current_log)
        hidden_dims.append(max(int(dim_val), 2))
        
    return hidden_dims


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1) 
        
        log_pt = log_probs.gather(1, labels.view(-1, 1)).view(-1)
        
        pt = log_pt.exp()
        focal_term = (-torch.expm1(log_pt)) ** self.gamma
        
        ce_loss = -log_pt
        
        loss = focal_loss = focal_term * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, labels)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DenoisingHeader(nn.Module):
    def __init__(self,
                 denoising_cfg,
                 dataset_name,     # "DVSCLEAN" or "EDKOGTL"
                 in_channels):
        super().__init__()
        self.dataset_name = dataset_name.upper()
        self.denoising_head_cfg = denoising_cfg[self.dataset_name]
        self.in_channels = in_channels
        
        self.num_classes = 2
        self.dropout = self.denoising_head_cfg.dropout
        
        self.if_hierarchical = self.denoising_head_cfg.if_hierarchical

        self.gnn_layers = nn.ModuleList()
        self.act_norm_layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        if self.if_hierarchical:
            self._build_gnn_backbone()
            classifier_input_dim = self.denoising_head_cfg.channels_list[-1]
        else:
            classifier_input_dim = in_channels

        hidden_dims = get_power_of_2_hidden_dims(classifier_input_dim, self.num_classes, 3)

        self.classifier = nn.Sequential(
            # nn.RMSNorm(classifier_input_dim),
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),

            # nn.RMSNorm(hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),

            # nn.RMSNorm(hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.Linear(hidden_dims[1], self.num_classes)
        )

        loss_cfg = self.denoising_head_cfg.get('loss', {})
        loss_type = loss_cfg.get('type', 'cross_entropy').lower()
        
        if loss_type == 'focal_loss':
            alpha = loss_cfg.alpha
            gamma = loss_cfg.gamma
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            smoothing = loss_cfg.get('label_smoothing', 0.0)
            self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def _build_gnn_backbone(self):
        hierarchical_params = self.denoising_head_cfg.hierarchical_params
        self.channels_list = self.denoising_head_cfg.channels_list
        self.num_stages = len(self.channels_list)
        
        gnn_type = hierarchical_params.gnn_type.upper()
        gnn_dropout = hierarchical_params.gnn_dropout
        gat_heads = hierarchical_params.gat_heads

        current_dim_for_gnn_input = self.in_channels
        
        for i in range(self.num_stages):
            gnn_output_dim_for_stage = self.channels_list[i]

            # --- a. GNN 层 ---
            if gnn_type == 'GAT':
                conv = GATConv(current_dim_for_gnn_input, gnn_output_dim_for_stage,
                               heads=gat_heads, dropout=gnn_dropout, concat=False)
            elif gnn_type == 'SAGE':
                conv = SAGEConv(current_dim_for_gnn_input, gnn_output_dim_for_stage)
            elif gnn_type == 'GCN':
                conv = GCNConv(current_dim_for_gnn_input, gnn_output_dim_for_stage)
            else:
                conv = GCNConv(current_dim_for_gnn_input, gnn_output_dim_for_stage)
            self.gnn_layers.append(conv)

            self.act_norm_layers.append(nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(gnn_output_dim_for_stage),
                nn.Dropout(p=gnn_dropout)
            ))

            if current_dim_for_gnn_input == gnn_output_dim_for_stage:
                self.skip_projections.append(nn.Identity())
            else:
                self.skip_projections.append(
                    nn.Linear(current_dim_for_gnn_input, gnn_output_dim_for_stage)
                )
            
            current_dim_for_gnn_input = gnn_output_dim_for_stage


    def forward(self, data: Batch):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        labels = data.denoising_label
        if labels.dtype != torch.long:
            labels = labels.long()

        if self.if_hierarchical:
            for i in range(self.num_stages):
                x_identity = x 

                x_conv = self.gnn_layers[i](x, edge_index)
                
                x_processed_gnn = self.act_norm_layers[i](x_conv)

                x_identity_projected = self.skip_projections[i](x_identity)
                x = x_processed_gnn + x_identity_projected
        
        logits = self.classifier(x) # [total_nodes, 2]

        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=-1) # [total_nodes]
        
        tp_tensor = (preds == 0) & (labels == 0)
        fp_tensor = (preds == 0) & (labels == 1)
        tn_tensor = (preds == 1) & (labels == 1)
        fn_tensor = (preds == 1) & (labels == 0)
        
        tp_count = tp_tensor.sum()
        fp_count = fp_tensor.sum()
        tn_count = tn_tensor.sum()
        fn_count = fn_tensor.sum()

        total_nodes = tp_count + fp_count + tn_count + fn_count
        acc = (tp_count + tn_count) / total_nodes if total_nodes > 0 else torch.tensor(0.0, device=logits.device)

        eps = torch.finfo(x.dtype).eps
        snr_ratio = tp_count.float() / (fp_count.float() + eps)
        snr = 10.0 * torch.log10(snr_ratio + eps)

        total_signal = tp_count + fn_count
        # total_noise = fp_count + tn_count
        recall = tp_count.float() / (total_signal.float() + eps)

        recall_threshold = 0.20

        if recall < recall_threshold:
            valid_snr = torch.tensor(-100.0, device=logits.device)
        else:
            valid_snr = snr

        output_dict = {
            'loss': loss, 
            'acc': acc, 
            'logits': logits,
            'preds': preds,
            'snr': snr,
            'recall': recall,
            'valid_snr': valid_snr,

            'tp': tp_count,
            'fp': fp_count,
            'fn': fn_count, 
            'tn': tn_count
        }
        return output_dict
        

