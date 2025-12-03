import torch.nn as nn
from eg_mamba.modules.multi_mamba import MultiMamba
from eg_mamba.modules.utils.processing_timer import ProcessingTimer


class NodesMamba(nn.Module):
    def __init__(self, 
                 d_model,
                 device,
                 d_state=64,
                 d_conv=4,
                 expand=2,
                 mamba_type="mamba",
                 num_splits=3,
                 reorder_type="three",
                 copy_parts_aggr_type="mean", # "max", "min", "sum", "mean", "learned"
                 num_loops=1,
                 if_use_residual=True,
                 residual_type="concat", # "simple_sum", "concat", "gate", "proj_sum", "none"
                 ):
        super().__init__()
        self.nodes_mamba = MultiMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mamba_type=mamba_type,
            num_splits=num_splits,
            reorder_type=reorder_type,
            copy_parts_aggr_type=copy_parts_aggr_type,
            num_loops=num_loops,
            if_use_residual=if_use_residual,
            residual_type=residual_type,
            device=device,
        )
        self.timer = ProcessingTimer(mode="gpu")

    def forward(self, list_data):
        """
        list_data: [(coords_0, nodes_0, comps_0, indices_0), ...]
        """
        self.timer.start()
        coords_final, nodes_final, comps_final = self.nodes_mamba(list_data) # [N, 3], [N, D], [N,]
        _, gpu_time = self.timer.stop()

        return coords_final, nodes_final, comps_final, gpu_time

