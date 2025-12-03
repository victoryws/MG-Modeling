import torch
import torch.nn as nn
from eg_mamba.modules.multi_mamba import MultiMamba
from eg_mamba.modules.utils.processing_timer import ProcessingTimer
from eg_mamba.modules.mamba_utils import compute_aggregation_batch, split_and_reorder_coords_nodes_comps
from eg_mamba.modules.flexible_gnn import FlexibleGNN
from torch_geometric.data import Data


class ComponentsNet(nn.Module):
    def __init__(self, 
                 d_model,
                 d_state=128,
                 d_conv=4,
                 expand=2,
                 mamba_type="mamba",
                 num_splits=1,
                 reorder_type="three",
                 copy_parts_aggr_type="learned", # "max", "min", "sum", "mean", "learned"
                 num_loops=1,
                 if_use_residual=True,
                 residual_type="gate", # "simple_sum", "concat", "gate", "proj_sum", "none"
                 if_use_compsgnn=False,
                 comps_gnn_type='GAT',
                 comps_coords_aggr_type='xyt_mean', # 'xyt_mean', 't_min', 't_max', 'xytsum_min', 'xytsum_max'
                 comps_feats_aggr_type='sum', # 'min', 'max', 'sum', 'mean'
                 device="cuda",
                 ):
        super().__init__()
        self.num_splits = num_splits
        self.reorder_type = reorder_type
        self.if_use_compsgnn = if_use_compsgnn
        self.comps_coords_aggr_type = comps_coords_aggr_type
        self.comps_feats_aggr_type = comps_feats_aggr_type

        if self.comps_feats_aggr_type == 'sum':
            self.intensity_mlp = SumNormWithLinearIntensity(dim=d_model)

        if self.if_use_compsgnn:
            self.components_gat = ComponentsGNN(
                dim=d_model,
                num_stages=num_loops,
                gnn_type=comps_gnn_type,
            )
        else:
            self.components_mamba = MultiMamba(
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

    def forward(self, coords, nodes, comps):
        """
            coords: [N, 3]
            nodes: [N, D]
            comps: [N,]
        """
        aggr_coords, aggr_nodes, aggr_comps, perm = compute_aggregation_batch(
            coords, 
            nodes, 
            comps, 
            coords_operation=self.comps_coords_aggr_type,
            nodes_operation=self.comps_feats_aggr_type,)
        
        if self.comps_feats_aggr_type == 'sum':
            aggr_nodes = self.intensity_mlp(aggr_nodes)
        
        if not self.if_use_compsgnn:
            list_data = split_and_reorder_coords_nodes_comps(
                aggr_coords, aggr_nodes, aggr_comps,
                split_parts=self.num_splits, # 1
                reorder_type=self.reorder_type, # "three"
            )
            self.timer.start()
            coords_final, nodes_final, comps_final = self.components_mamba(list_data) # [aggr_N, 3], [aggr_N, D], [aggr_N,]
            _, gpu_time = self.timer.stop()

        else:
            self.timer.start()
            coords_final, nodes_final, comps_final = self.components_gat(aggr_coords, aggr_nodes, aggr_comps)
            _, gpu_time = self.timer.stop()
        
        return coords_final, nodes_final, comps_final, gpu_time


class SumNormWithLinearIntensity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.norm = nn.RMSNorm(dim)
        self.norm = nn.LayerNorm(dim)
        
        self.intensity_proj = nn.Linear(1, dim)
        
        nn.init.zeros_(self.intensity_proj.weight)
        nn.init.zeros_(self.intensity_proj.bias)

    def forward(self, x_sum):
        magnitude = torch.norm(x_sum, p=2, dim=-1, keepdim=True)
        log_mag = torch.log(magnitude + torch.finfo(x_sum.dtype).eps)
        
        x_normalized = self.norm(x_sum)
        dynamic_bias = self.intensity_proj(log_mag)
        
        return x_normalized + dynamic_bias
    

class ComponentsGNN(nn.Module):
    def __init__(self, dim, num_stages, gnn_type='GAT'):
        super().__init__()
        self.num_stages = num_stages
        
        self.compsgat_list = nn.ModuleList([
                FlexibleGNN(
                    input_dim=dim, 
                    output_dim=dim, 
                    if_use_gnn=True,
                    num_layers=1, 
                    gnn_type=gnn_type,
                ) for _ in range(num_stages)
            ])
        
        # self.act = nn.SiLU()
        # self.norm = nn.LayerNorm(dim)

    def construct_graph(self, coords, nodes, comps):
        device = coords.device
        num_nodes = nodes.size(0)
        indices = torch.arange(num_nodes, device=device)
        row, col = torch.meshgrid(indices, indices, indexing="ij")
        edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)
        comps_graph = Data(x=nodes, edge_index=edge_index, coords=coords, component_labels=comps)
        return comps_graph
    
    def forward(self, coords, nodes, comps):
        # Step 1: Construct the graph
        comps_graph = self.construct_graph(coords, nodes, comps)
        
        # Step 2: Apply GAT layers
        for layer in self.compsgat_list:
            comps_graph = layer(comps_graph)
        
        return comps_graph.coords, comps_graph.x, comps_graph.component_labels
        
        

        








# import torch
# import torch.nn as nn
# from torch_scatter import scatter_mean
# from mamba_ssm import Mamba, Mamba2
# from torch_geometric.data import Data, Batch
# from eg_mamba.modules.flexible_gnn import FlexibleGNN
# from eg_mamba.modules.mamba_utils import compute_aggregation_batch, split_and_reorder_coords_nodes_comps
# from eg_mamba.modules.graph_utils import get_activation, get_normalization



# class ComponentsMamba(nn.Module):
#     def __init__(self, 
#                  dim, 
#                  split_parts=1,
#                  reorder_type="three", 
#                  aggregation='sum_max', 
#                  num_stages=1,
#                  if_adp_multiweights_nn=True,
#                  if_additional_gat=False):
#         super().__init__()
#         self.num_stages = num_stages
#         self.aggregation = aggregation
#         reorder_types = {"three": 3, "six": 6, "one": 1, "two": 2}
#         if reorder_type not in reorder_types:
#             raise ValueError(f"Invalid reorder_type: {reorder_type}")
#         self.num_parts = reorder_types[reorder_type]
#         self.reorder_type = reorder_type
#         self.split_parts = split_parts
        
#         self.compsmamba_list = nn.ModuleList()
#         for _ in range(num_stages):
#             # stage = nn.ModuleList([Mamba(dim) for _ in range(self.num_parts)])
#             stage = nn.ModuleList(
#                 [Mamba2(d_model=dim,
#                         expand=4,
#                         headdim=dim // 2,) # "使用 Mamba2 时 d_model * expand / headdim 必须是 8 的倍数"
#                  for _ in range(self.num_parts)]
#             )
#             self.compsmamba_list.append(stage)
        
#         # multi_weighted_aggregation NN
#         if if_adp_multiweights_nn and self.num_parts > 1:
#             self.multi_weighted_nn = MultiWeightsNet(dim, if_adp_multiweights_nn)
#         else: # False or num_parts=1
#             self.multi_weighted_nn = MultiWeightsNet(dim, if_adp_multiweights_nn=False)

#         self.if_additional_gat = if_additional_gat
#         if self.if_additional_gat:
#             self.additional_gat_layer = AdditionalGATLayer(dim=dim)
        
#         self.act = get_activation("silu")
#         self.norm = get_normalization("batch", dim)

#     def _sort_by_time_column(self, coords, nodes, comps):
#         sorted_indices = coords[:, 2].argsort()  # Sort by time column
#         return (coords[sorted_indices], nodes[sorted_indices], comps[sorted_indices])
    
#     def sort_all_by_time(self, sorted_aggr_results_list):
#         # Process each tuple in the list by sorting based on the time column
#         return [self._sort_by_time_column(coords, nodes, comps) for coords, nodes, comps in sorted_aggr_results_list]
    
#     def multi_direction_weighted_sum(self, input_list):
#         probs = self.multi_weighted_nn(input_list)  # Shape: [3/6, 1]
        
#         # Step 1: Extract nodes tensors
#         nodes_tensors_list = [item[1] for item in input_list]  # Extracting the second element (nodes) from each tuple
#         coords, _, comps = input_list[0]

#         # Step 2: Stack nodes tensors along a new dimension
#         stacked_nodes = torch.stack(nodes_tensors_list, dim=0)  # Shape: [3/6, n, dim]
        
#         # Step 3: Adjust probs to match dimensions for broadcasting
#         adjusted_probs = probs.view(-1, 1, 1)  # Shape: [3/6, 1, 1]
        
#         # Step 4: Apply weighted sum using broadcasting
#         weighted_nodes = stacked_nodes * adjusted_probs  # Broadcasting the multiplication
#         sum_nodes = torch.sum(weighted_nodes, dim=0)  # Summing across the first dimension (the 3/6 groups)

#         # assert torch.isnan(sum_nodes).any() == False, "sum_nodes contains NaN values."
#         # assert torch.isinf(sum_nodes).any() == False, "sum_nodes contains inf values."
#         # if torch.isnan(sum_nodes).any() or torch.isinf(sum_nodes).any():
#         #     print("sum_nodes contains NaN or inf values.")
#         #     print("sum_nodes:", sum_nodes)

#         # sum_nodes = sum_nodes.clamp(min=-1e6, max=1e6)  # Prevent excessively large values

#         # Return as a list containing a single tuple
#         return [(coords, sum_nodes, comps)]  # cuda 0

#     # 先整合成一个大batch，然后再处理
#     def _prepare_data_list(self, sorted_results_list):
#         # Step 0: Concatenate all coords, nodes, and comps from the list
#         all_coords = torch.cat([item[0] for item in sorted_results_list], dim=0) # shape: torch.Size([77709, 3])
#         all_nodes = torch.cat([item[1] for item in sorted_results_list], dim=0) # shape: torch.Size([77709, 8])
#         all_comps = torch.cat([item[2] for item in sorted_results_list], dim=0) # shape: torch.Size([77709])

#         # Step 1: Aggregate the concatenated data
#         # aggr_coords, aggr_nodes, aggr_comps, perm = compute_aggregation_batch(all_coords, all_nodes, all_comps, operation=self.aggregation) # aggr_coords: torch.Size([16, 3]), aggr_nodes: torch.Size([16, 8]), aggr_comps: torch.Size([16])
#         aggr_coords, aggr_nodes, aggr_comps, perm = compute_aggregation_batch(
#             all_coords, all_nodes, all_comps, operation=self.aggregation
#         )  # [M, 3], [M, 8], [M]
        
#         # Step 2: If needed, split and reorder aggregated results
#         sorted_aggr_results_list = split_and_reorder_coords_nodes_comps(
#             aggr_coords, aggr_nodes, aggr_comps, 
#             split_parts=self.split_parts, 
#             reorder_type=self.reorder_type, 
#             # split_parts=1 和 reorder_type='three' 保证了是同一个部分被划分为了三个方向 所以后面可以multi_direction_weighted_sum求和
#         )
#         return sorted_aggr_results_list
    
#     def forward(self, updated_sorted_results_list):
#         # Prepare the data for the first stage
#         # print("CompsMamba中的updated_sorted_results_list:", updated_sorted_results_list)
#         sorted_aggr_results_list = self._prepare_data_list(updated_sorted_results_list)
#         # print("CompsMamba中的sorted_aggr_results_list:", sorted_aggr_results_list)

#         # Step 4: Process each part through Mamba
#         for stage in self.compsmamba_list:
#             for i in range(self.num_parts):
#                 sorted_aggr_coords, sorted_aggr_nodes, sorted_aggr_comps = sorted_aggr_results_list[i]
                
#                 if sorted_aggr_nodes.dim() == 2:
#                     sorted_aggr_nodes = sorted_aggr_nodes.unsqueeze(0)  # Add a batch dimension if needed
#                     assert sorted_aggr_nodes.dim() == 3, f"Input tensor must be 3D (batch_size, sequence_length, dimension), got shape {sorted_aggr_nodes.shape}"
                
#                 nodes_i_processed = stage[i](sorted_aggr_nodes)  # Process nodes
#                 nodes_i_processed = nodes_i_processed.squeeze(0)  # Remove the batch dimension
#                 nodes_i_processed = self.act(nodes_i_processed)  # Apply activation
#                 nodes_i_processed = self.norm(nodes_i_processed)

#                 # if torch.isnan(nodes_i_processed).any() or torch.isinf(nodes_i_processed).any():
#                 #     print("nodes_i_processed contains NaN or inf values.")
#                 #     print("nodes_i_processed:", nodes_i_processed)
#                 # # 通过当前 stage 中的第 i 个 Mamba 模块处理节点
#                 # sorted_aggr_nodes = stage[i](sorted_aggr_nodes)

#                 # # act + norm
#                 # aggr_nodes = nn.SiLU()(aggr_nodes)
#                 # aggr_nodes = nn.LayerNorm(aggr_nodes.size(-1))(aggr_nodes)

#                 sorted_aggr_results_list[i] = (sorted_aggr_coords, nodes_i_processed, sorted_aggr_comps) # Update
#         # 包含了同一组数据的三向/六向的处理结果

#         # 根据probs融合多向的处理结果
#         sort_by_time_sorted_aggr_results_list = self.sort_all_by_time(sorted_aggr_results_list)
#         # aggr_results_list = self.multi_direction_weighted_sum(sort_by_time_sorted_aggr_results_list)

#         if self.if_additional_gat:
#             gat_out = self.additional_gat_layer(
#                             self.multi_direction_weighted_sum(sort_by_time_sorted_aggr_results_list)
#                             )
#             # print("gat_out Done!")
#             return gat_out
#         else:
#             # print("not_gat_out start!")
#             not_gat_out = self.multi_direction_weighted_sum(sort_by_time_sorted_aggr_results_list)  # [(coords, sum_nodes, comps)]
#             # print("not_gat_out Done!")
#             return not_gat_out

'''先处理每个部分, 然后再整合成一个大batch
    def _prepare_data_list(self, updated_sorted_results_list): # 这是先处理每个部分，然后再整合成一个大batch
        batch_aggr_coords_list = []
        batch_aggr_nodes_list = []
        batch_aggr_comps_list = []

        # Step 1: Aggregate each part
        for coords, nodes, comps in updated_sorted_results_list:
            aggr_coords, aggr_nodes, aggr_comps, perm = compute_aggregation_batch(coords, nodes, comps, operation=self.aggregation)
            batch_aggr_coords_list.append(aggr_coords)
            batch_aggr_nodes_list.append(aggr_nodes)
            batch_aggr_comps_list.append(aggr_comps)
        
        # Step 2: Concatenate all aggregates
        batch_aggr_coords_tensor = torch.cat(batch_aggr_coords_list, dim=0)
        batch_aggr_nodes_tensor = torch.cat(batch_aggr_nodes_list, dim=0)
        batch_aggr_comps_tensor = torch.cat(batch_aggr_comps_list, dim=0)

        # Step 3: Split and reorder using a helper function
        sorted_aggr_results_list = split_and_reorder_coords_nodes_comps(batch_aggr_coords_tensor, batch_aggr_nodes_tensor, batch_aggr_comps_tensor, split_parts=1, reorder_type=self.reorder_type)

        return sorted_aggr_results_list'''




# class AdditionalGATLayer(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # self.gatconv = GATConv(dim, dim, heads=2, concat=False)
#         self.gatconv = FlexibleGNN(
#             input_dim=dim,
#             output_dim=dim,
#             if_use_gnn=True,
#             gnn_type='GAT',
#         )
#         self.act = get_activation("silu")
#         self.norm = get_normalization("layer", dim)

#     def forward(self, input_list):  # [(coords, sum_nodes, comps)]
#         x = input_list[0][1]
#         device = x.device
#         num_nodes = x.size(0)

#         # 创建全连接图，但避免内存过大
#         edge_index = torch.combinations(torch.arange(num_nodes, device=device), r=2).t().contiguous()
#         edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)  # Make it undirected

#         graph = self.gatconv(Data(x=x, edge_index=edge_index))
#         x = self.act(graph.x)
#         x = self.norm(x)
#         return [(input_list[0][0], x, input_list[0][2])]



# class MultiWeightsNet(nn.Module):
#     def __init__(self, dim, if_adp_multiweights_nn):  # dim=mamba_dim
#         super().__init__()
#         self.if_adp_multiweights_nn = if_adp_multiweights_nn
#         if if_adp_multiweights_nn:
#             self.expand_layer = nn.Sequential(
#                 nn.Linear(dim, dim * 2),
#                 nn.SiLU(),
#                 nn.Dropout(0.2),
#                 nn.LayerNorm(dim * 2)
#             )
#             self.prob_layer = nn.Sequential(
#                 nn.Linear(dim * 2, 1),
#                 nn.Softmax(dim=0)
#             )
#             self._init_weights()

#     def _init_weights(self):
#         if self.if_adp_multiweights_nn:
#             for m in self.expand_layer:
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight)
#                     nn.init.zeros_(m.bias)
#             for m in self.prob_layer:
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight)
#                     nn.init.zeros_(m.bias)

#     def _prepare_tensor(self, sorted_aggr_results_list):
#         batch_nodes_list = []
#         group_indices_list = []

#         for i, (_, nodes, _) in enumerate(sorted_aggr_results_list):
#             batch_nodes_list.append(nodes)
#             group_indices_list.append(torch.full((nodes.size(0),), i, device=nodes.device))

#         batch_nodes = torch.cat(batch_nodes_list, dim=0)
#         group_indices = torch.cat(group_indices_list, dim=0)

#         return batch_nodes, group_indices

#     def forward(self, sorted_aggr_results_list):
#         if self.if_adp_multiweights_nn:
#             batch_nodes, group_indices = self._prepare_tensor(sorted_aggr_results_list)

#             expanded_nodes = self.expand_layer(batch_nodes)

#             group_means = scatter_mean(expanded_nodes, group_indices, dim=0)

#             probs = self.prob_layer(group_means)
#             probs = probs.clamp(min=1e-6, max=1.0)  # Ensure numerical stability
#         else:
#             type_length = len(sorted_aggr_results_list)
#             # print(f"type_length: {type_length}")
#             device = sorted_aggr_results_list[0][0].device
#             probs = torch.ones(type_length, 1, device=device) / type_length

#         return probs  # 返回的是每个组的权重, 是[3,1]或[6,1]的形式

