import torch
import torch.nn as nn

from eg_mamba.modules.hybrid_multi_event_graph_mamba import HybridMultiEventGraphMamba


class BackboneFeaturer(nn.Module):
    def __init__(self, 
                 backbone_cfg,
                 batch_size,
                 device,
                 ):
        super().__init__()
        if backbone_cfg.dataset_name.lower() in ['gen1', 'gen4']:
            self.task_type = 'detection'
        elif backbone_cfg.dataset_name.lower() in ['ncars']:
            self.task_type = 'classification'
        elif backbone_cfg.dataset_name.lower() in ['thueact50chl']:
            self.task_type = 'action'
        elif backbone_cfg.dataset_name.lower() in ['dvsclean', 'edkogtl']:
            self.task_type = 'denoising'
        else:
            raise ValueError("Invalid backbone_cfg.dataset_name. Choose 'gen1', 'gen4', 'ncars', 'thueact50chl'.")
        self.backbone = HybridMultiEventGraphMamba(backbone_cfg, batch_size, device)

    def forward(self, ev_gdata):
        if self.task_type in ['detection', 'tracking']:
            backbone_feats_dict, time_info_dict = self.backbone(ev_gdata)
            return backbone_feats_dict, time_info_dict # Dict[int, torch.Tensor]
        elif self.task_type in ['classification', 'action', 'denoising']:
            # ev_gdata is a torch_geometric.data.Data object
            output_graph, time_info_dict = self.backbone(ev_gdata) # torch_geometric.data.Data
            return output_graph, time_info_dict
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")