import torch
import torch.nn as nn

from eg_mamba.modules.eg_nodes_mamba import NodesMamba
from eg_mamba.modules.eg_comps_net import ComponentsNet
from eg_mamba.modules.aggr_fusion import AggregationFusion
from eg_mamba.modules.coords_perturbation import UniqueCoordinatesGenerator, FlexibleUniqueCoordinatesGenerator
from eg_mamba.modules.mamba_utils import split_and_reorder_coords_nodes_comps


class NodesComponentsBlock(nn.Module):
    def __init__(self, 
                 # nn modules
                 feat_dim=8,

                 if_update_unique_coords=False,
                 
                 nodes_stages=1,
                 nodes_split_parts=3, # or 6 or 1
                 nodes_reorder_type="three", # or "six" or 'one
                 nodes_mamba_type="mamba2", # or "mamba2"
                 nodes_residual_type="sigmoid", # "simple_sum", "concat", "gate", "proj_sum", "sigmoid", "none"
                 nodes_copy_parts_aggr_type="mean", # "max", "min", "sum", "mean", "learned"

                 comps_stages=1,
                 comps_coords_aggr_type='xyt_mean', # 'xyt_mean', 't_min', 't_max', 'xytsum_min', 'xytsum_max'
                 comps_feats_aggr_type='sum', # 'min', 'max', 'sum', 'mean'
                 comps_split_parts=1, # or 6 or 1
                 comps_reorder_type="three", # or "six" or 'one
                 comps_mamba_type="mamba2", # or "mamba2"
                 comps_residual_type="sigmoid", # "simple_sum", "concat", "gate", "proj_sum", "sigmoid", "none"
                 comps_copy_parts_aggr_type="learned", # "max", "min", "sum", "mean", "learned"
                 if_comps_gnn=False,
                 comps_gnn_type="GAT", # 'GCN', 'GAT', 'SAGE', 'GIN', 'GraphConv', 'ChebConv', 'SGConv'

                 nodes_comps_fusion_type="sum", # sum, concat, gate, mlp_res
                 
                 device='cuda',

                 
                 
                 ):
        super().__init__()
        self.unique_coords_generator = FlexibleUniqueCoordinatesGenerator(dim=feat_dim, 
                                                                  if_update_coords=if_update_unique_coords)

        self.nodes_split_parts = nodes_split_parts
        self.nodes_reorder_type = nodes_reorder_type

        self.comps_split_parts = comps_split_parts
        self.comps_reorder_type = comps_reorder_type
        
        self.nodes_mamba = NodesMamba(d_model=feat_dim, 
                                      d_state=128,
                                      d_conv=4,
                                      expand=2,
                                      mamba_type=nodes_mamba_type, # mamba2
                                      num_splits=nodes_split_parts,
                                      reorder_type=self.nodes_reorder_type, 
                                      copy_parts_aggr_type=nodes_copy_parts_aggr_type, # "max", "min", "sum", "mean", "learned"
                                      num_loops=nodes_stages,
                                      if_use_residual=True,
                                      residual_type=nodes_residual_type, # "simple_sum", "concat", "gate", "proj_sum", "none"
                                      device=device,)
        
        self.comps_net = ComponentsNet(d_model=feat_dim,
                                       d_state=128,
                                       mamba_type=comps_mamba_type, # mamba2
                                       num_splits=comps_split_parts,
                                       reorder_type=self.comps_reorder_type, 
                                       copy_parts_aggr_type=comps_copy_parts_aggr_type, # "max", "min", "sum", "mean", "learned"
                                       num_loops=comps_stages,
                                       if_use_compsgnn=if_comps_gnn,
                                       comps_gnn_type=comps_gnn_type,
                                       if_use_residual=True,
                                       residual_type=comps_residual_type, # "simple_sum", "concat", "gate", "proj_sum", "none"
                                       comps_coords_aggr_type=comps_coords_aggr_type,
                                       comps_feats_aggr_type=comps_feats_aggr_type,
                                       device=device)

        self.aggr_fusion = AggregationFusion(fusion_type=nodes_comps_fusion_type, # sum, concat, gate, mlp_res
                                             node_dim=feat_dim)

        
    def forward(self, input_graph):
        # --------------- Unique Coordinates Generation (including perturbation) -----------------
        unique_coords = self.unique_coords_generator(input_graph)

        splited_sorted_list = split_and_reorder_coords_nodes_comps(
                unique_coords, 
                input_graph.x, 
                input_graph.component_labels, 
                split_parts=self.nodes_split_parts, 
                reorder_type=self.nodes_reorder_type,
                )

        # ---------------------- Nodes Mamba ----------------------
        nm_coords, nm_nodes, nm_comps, nm_gpu_time = self.nodes_mamba(splited_sorted_list)

        # ---------------------- Components Mamba / Components GAT ----------------------
        cn_aggr_coords, cn_aggr_nodes, cn_aggr_comps, cn_gpu_time = self.comps_net(nm_coords, nm_nodes, nm_comps)
        
        fused_coords, fused_nodes, fused_comps, fusion_gpu_time = self.aggr_fusion(
            nm_coords, nm_nodes, nm_comps, 
            cn_aggr_coords, cn_aggr_nodes, cn_aggr_comps
        )

        input_graph.x = fused_nodes
        input_graph.component_labels = fused_comps
        time_dict = {
            # "sum": nm_gpu_time + cn_gpu_time + fusion_gpu_time,
            "nodes": nm_gpu_time,
            "comps": cn_gpu_time,
            "fusion": fusion_gpu_time
        }
        return input_graph, fused_coords, time_dict




from torch_scatter import scatter_mean

class GraphFusionModule(nn.Module):
    def __init__(self, node_dim, if_adp_gating=True):
        super().__init__()
        if if_adp_gating:
            self.gating_layer = nn.Sequential(
                nn.Linear(node_dim*2, node_dim*4),
                nn.SiLU(),
                nn.RMSNorm(node_dim*4),
                nn.Linear(node_dim*4, 2),
                nn.Softmax(dim=-1)
            )
        else:
            None
        self.if_adp_gating = if_adp_gating

    def adaptive_gating_layer(self, orig_nodes, aggr_nodes):
        if self.if_adp_gating:
            combined_nodes = torch.cat([orig_nodes, aggr_nodes], dim=-1)
            gating_values = self.gating_layer(combined_nodes)
            fused_nodes = orig_nodes * gating_values[:, 0].unsqueeze(-1) + aggr_nodes * gating_values[:, 1].unsqueeze(-1)
        else:
            fused_nodes = orig_nodes + aggr_nodes

        return fused_nodes

    def forward(self, orig_graph, aggr_results_list):
        orig_nodes, orig_comps = orig_graph.x, orig_graph.component_labels
        aggr_coords, aggr_nodes, aggr_comps = aggr_results_list[0]

        unique_comps, inverse_indices = torch.unique(aggr_comps, return_inverse=True, sorted=True)
        mean_aggr_nodes = scatter_mean(aggr_nodes, inverse_indices, dim=0)

        # Step 2: 
        mapped_indices = torch.searchsorted(unique_comps, orig_comps)
        corresponding_aggr_nodes = mean_aggr_nodes[mapped_indices]

        # Step 3: 
        fused_nodes = self.adaptive_gating_layer(orig_nodes, corresponding_aggr_nodes)

        orig_graph.x = fused_nodes
        return orig_graph


