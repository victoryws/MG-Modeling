import torch
import torch.nn as nn

from typing import Optional, Dict
from omegaconf import OmegaConf


from tasks.detection.rtdetr_header import RTDETRDetector
from tasks.detection.yolox_header import YoloXDetector
from tasks.classification.cls_header import ClassificationHeader
from tasks.action.action_header import ActionHeader
from tasks.denoising.denoising_header import DenoisingHeader


class TaskConstructor(nn.Module):
    def __init__(self, 
                 backbone_cfg,
                 task_cfg,
                 detection_cfg=None,
                 classification_cfg=None,
                 action_cfg=None,
                 denoising_cfg=None,
                 ):
        super().__init__()
        # self.backbone_cfg = backbone_cfg
        self.dataset_name = backbone_cfg.dataset_name.upper()
        if self.dataset_name in ['GEN1', 'GEN4']:
            self.task_type = 'detection'
        
        elif self.dataset_name in ['NCARS']:
            self.task_type = 'classification'
        
        elif self.dataset_name in ['THUEACT50CHL', 'DVSACTION']:
            self.task_type = 'action'
        
        elif self.dataset_name in ['DVSCLEAN', 'EDKOGTL']:
            self.task_type = 'denoising'
        
        else:
            raise ValueError("Invalid backbone_cfg.dataset_name. Choose 'gen1' or 'gen4' or 'ncars' or 'dvsclean' or 'edkogtl'.")

        self.task_cfg = task_cfg
        self.detection_cfg = detection_cfg
        self.denoising_cfg = denoising_cfg

        if self.task_type not in task_cfg:
            raise ValueError(f"Task type '{self.task_type}' not found in task configuration.")
        task_specific_cfg = task_cfg[self.task_type]  # Get the configuration specific to the task
        head_type = task_specific_cfg['head']  # E.g., detection:'rtdetr', 'yolox', classification:'fusion', tracking:'odtrack'

        # Initialize detector based on head
        if self.task_type == 'detection':
            if head_type not in detection_cfg: # 'rtdetr', 'yolox'
                raise ValueError(f"Head '{head_type}' not found in detection_cfg.")
            det_head_cfg = detection_cfg[head_type]
            if head_type == 'rtdetr':
                self.detector = RTDETRDetector(det_head_cfg, dataset_name=backbone_cfg.dataset_name)
            elif head_type == 'yolox':
                self.detector = YoloXDetector(det_head_cfg, dataset_name=backbone_cfg.dataset_name)
            else:
                raise ValueError(f"Unknown detection head: {head_type}")

        elif self.task_type == 'classification':
            if head_type == 'multifusion':
                cls_head_cfg = classification_cfg[head_type]
                self.classifier = ClassificationHeader(cls_head_cfg)
            else:
                raise ValueError(f"Unknown classification head: {head_type}")
            
        elif self.task_type == 'action':
            head_upper_type = head_type.upper()
            if head_upper_type not in action_cfg:
                raise ValueError(f"Head config of '{head_upper_type}' not found in action_cfg.")
            
            self.action_header = ActionHeader(
                action_cfg, 
                head_type.upper(), 
                in_channels=backbone_cfg[self.dataset_name].feat_dims[-1],
                )
        
        elif self.task_type == 'denoising':
            head_upper_type = head_type.upper()
            if head_upper_type not in denoising_cfg:
                raise ValueError(f"Head config of '{head_upper_type}' not found in denoising_cfg.")
            
            self.denoising_header = DenoisingHeader(
                denoising_cfg, 
                self.dataset_name, 
                in_channels=backbone_cfg[self.dataset_name].feat_dims[-1],
                )
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")


    def task_header(self, feats_dict_or_graph, targets): # Dict[int, torch.Tensor]
        if self.task_type == 'detection':
            return self.detector(feats_dict_or_graph, targets)
        
        elif self.task_type == 'classification':
            return self.classifier(feats_dict_or_graph, targets)
        
        elif self.task_type == 'action':
            return self.action_header(feats_dict_or_graph, targets)

        elif self.task_type == 'denoising':
            return self.denoising_header(feats_dict_or_graph)
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def forward(self, 
                feats_dict_or_graph, 
                targets: Optional[torch.Tensor] = None
                ):
        # feats_dict_or_graph = self.forward_backbone(ev_data)
        return self.task_header(feats_dict_or_graph, targets) 
