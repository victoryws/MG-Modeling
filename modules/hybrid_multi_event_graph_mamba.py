import torch
import torch.nn as nn

from eg_mamba.modules.flexible_gnn import FlexibleGNN
from eg_mamba.modules.graph_utils import GraphSampler
from eg_mamba.modules.graph_pe import GraphPositionalEncoding
from eg_mamba.modules.nodescomps_block import NodesComponentsBlock
from eg_mamba.modules.vision_reshaper import GraphToVisionReshaper
from eg_mamba.modules.utils.processing_timer import ProcessingTimer


class HybridMultiEventGraphMamba(nn.Module):
    def __init__(self, 
                 backbone_cfg,
                 batch_size,
                 device,
                #  num_workers,
                 ):
        super().__init__()
        self.dataset_name = backbone_cfg.dataset_name # 'gen1' or 'gen4' or ''
        
        if self.dataset_name.lower() in ['gen1', 'gen4']:
            self.task_type = 'detection'
        elif self.dataset_name.lower() in ['ncars']:
            self.task_type = 'classification'
        elif self.dataset_name.lower() in ['thueact50chl', 'dvsaction']:
            self.task_type = 'action'
        elif self.dataset_name.lower() in ['dvsclean', 'edkogtl']:
            self.task_type = 'denoising'
        else:
            raise ValueError(f"Invalid dataset_name: {self.dataset_name}. Invalid task_type: {self.task_type}.")

        backbone_spec_cfg = backbone_cfg[self.dataset_name.upper()]
        if self.task_type in ['detection', 'tracking']:
            if_gr2vis_feat_avg = backbone_spec_cfg.if_gr2vis_feat_avg
            if_gr2vis_additional = backbone_spec_cfg.if_gr2vis_additional
        
        feat_dims = backbone_spec_cfg.feat_dims # [8, 16, 32, 64]
        sample_ratios = backbone_spec_cfg.sample_ratios # [0.5, 1.0, 1.0, 1.0]
        sample_poolmodes = backbone_spec_cfg.sample_poolmodes # 'topk', 'sag', 'asa', 'uniform', 'random'

        graph_pe_cfg = backbone_spec_cfg.graph_pe
        if_use_pe_s = graph_pe_cfg.if_use_pe_s # [True, True, True, True]
        pe_type = graph_pe_cfg.pe_type # 'learned', 'sincos', 'laplacian'
        num_components = graph_pe_cfg.num_components # 16
        
        # if_update_gat=[False, False, False, False],

        nodes_split_parts = backbone_spec_cfg.nodes_split_parts # [3, 3, 3, 3]
        nodes_reorder_types = backbone_spec_cfg.nodes_reorder_types # ["three", "three", "three", "three"]
        nodes_residual_types = backbone_spec_cfg.nodes_residual_types # ["simple_sum", "simple_sum", "simple_sum", "simple_sum"]
        nodes_copy_parts_aggr_types = backbone_spec_cfg.nodes_copy_parts_aggr_types # ["mean", "mean", "mean", "mean"]
        nodes_mamba_type = backbone_spec_cfg.nodes_mamba_type # 'mamba2', # 'mamba' or 'mamba2'

        comps_split_parts = backbone_spec_cfg.comps_split_parts # [1, 1, 1, 1]
        comps_reorder_types = backbone_spec_cfg.comps_reorder_types # ["one", "three", "one", "three"]
        comps_copy_parts_aggr_types = backbone_spec_cfg.comps_copy_parts_aggr_types # ["learned", "learned", "learned", "learned"]
        comps_mamba_type = backbone_spec_cfg.comps_mamba_type # 'mamba', # 'mamba' or 'mamba2'

        if_update_unique_coords_s = backbone_spec_cfg.if_update_unique_coords_s # [True, False, False, False]
        nodes_stages = backbone_spec_cfg.nodes_stages # [1, 1, 1, 1], # epochs of nodes_mamba in each stage
        comps_stages = backbone_spec_cfg.comps_stages # [1, 1, 1, 1], # epochs of comps_mamba in each stage

        comps_coords_aggr_type = backbone_spec_cfg.comps_coords_aggr_type # 'xyt_mean', 't_min', 't_max', 'xytsum_min', 'xytsum_max'
        comps_feats_aggr_type = backbone_spec_cfg.comps_feats_aggr_type # 'min', 'max', 'sum', 'mean'
        comps_residual_types = backbone_spec_cfg.comps_residual_types # ["simple_sum", "simple_sum", "simple_sum", "simple_sum"]

        if_use_comps_gnns = backbone_spec_cfg.if_use_comps_gnns # [False, True, False, True],
        comps_gnn_types = backbone_spec_cfg.comps_gnn_types # ['GAT', 'GAT', 'GAT', 'GAT']

        per_in_gnns_cfg = backbone_spec_cfg.per_in_gnns
        per_out_gnns_cfg = backbone_spec_cfg.per_out_gnns

        nodes_comps_fusion_types = backbone_spec_cfg.nodes_comps_fusion_types # ['sum', 'sum', 'sum', 'sum']

        num_block_epochs = len(feat_dims) # 4

        self.per_in_gnns = nn.ModuleList(
                [FlexibleGNN(input_dim=4 if i == 0 else feat_dims[i-1], 
                             output_dim=feat_dims[i],
                             if_use_gnn=per_in_gnns_cfg.if_use_gnn[i],
                             num_layers=per_in_gnns_cfg.num_layers[i],
                             gnn_type=per_in_gnns_cfg.gnn_type[i],
                             activation=per_in_gnns_cfg.activation,
                             normalization=per_in_gnns_cfg.normalization[i],
                             if_use_skip_connection=per_in_gnns_cfg.if_use_skip_connection[i],
                            
                             if_use_dropout=per_in_gnns_cfg.if_use_dropout,
                             dropout=per_in_gnns_cfg.dropout,)
                for i in range(num_block_epochs)])

        self.graph_samplers = [GraphSampler(in_dim=feat_dims[i], 
                                            ratio=sample_ratios[i], 
                                            poolmode=sample_poolmodes[i]) 
                               for i in range(num_block_epochs)]

        self.graph_pe = nn.ModuleList([GraphPositionalEncoding(out_dim=feat_dims[i], 
                                                               if_use_pe=if_use_pe_s[i],
                                                               pe_type=pe_type,
                                                               num_components=num_components
                                                               ) 
                                        for i in range(num_block_epochs)])

        self.nodescomps_blocks = nn.ModuleList([
                NodesComponentsBlock(feat_dim=feat_dims[i],
                                     if_update_unique_coords=if_update_unique_coords_s[i],
                                     nodes_stages=nodes_stages[i],
                                     nodes_split_parts=nodes_split_parts[i],
                                     nodes_reorder_type=nodes_reorder_types[i],
                                     nodes_mamba_type=nodes_mamba_type, # 'mamba' or 'mamba2'
                                     nodes_residual_type=nodes_residual_types[i], # "simple_sum", "concat", "gate", "proj_sum", "none"
                                     nodes_copy_parts_aggr_type=nodes_copy_parts_aggr_types[i], # "max", "min", "sum", "mean", "learned"
                                     
                                     comps_stages=comps_stages[i],
                                     comps_coords_aggr_type=comps_coords_aggr_type, # 'xyt_mean', 't_min', 't_max', 'xytsum_min', 'xytsum_max'
                                     comps_feats_aggr_type=comps_feats_aggr_type, # 'min', 'max', 'sum', 'mean'
                                     comps_split_parts=comps_split_parts[i],
                                     comps_reorder_type=comps_reorder_types[i],
                                     comps_mamba_type=comps_mamba_type, # 'mamba' or 'mamba2'
                                     comps_residual_type=comps_residual_types[i], # "simple_sum", "concat", "gate", "proj_sum", "none"
                                     comps_copy_parts_aggr_type=comps_copy_parts_aggr_types[i], # "max", "min", "sum", "mean", "learned"
                                     
                                     if_comps_gnn=if_use_comps_gnns[i],
                                     comps_gnn_type=comps_gnn_types[i], # 'GCN','GAT','SAGE','GIN','GraphConv','ChebConv','SGConv'

                                     nodes_comps_fusion_type=nodes_comps_fusion_types[i], # 'sum', 'concat', 'gate', 'mlp_res'

                                     device=device,
                                    )
                for i in range(num_block_epochs)])
        
        self.per_out_gnns = nn.ModuleList(
                [FlexibleGNN(input_dim=feat_dims[i], 
                             output_dim=feat_dims[i],
                             if_use_gnn=per_out_gnns_cfg.if_use_gnn[i],
                             num_layers=per_out_gnns_cfg.num_layers[i],
                             gnn_type=per_out_gnns_cfg.gnn_type[i],
                             activation=per_out_gnns_cfg.activation,
                             normalization=per_out_gnns_cfg.normalization[i],
                             if_use_skip_connection=per_out_gnns_cfg.if_use_skip_connection[i],
                            
                             if_use_dropout=per_out_gnns_cfg.if_use_dropout,
                             dropout=per_out_gnns_cfg.dropout,)
                for i in range(num_block_epochs)])
        
        if self.task_type in ['detection', 'tracking']:
            self.if_graph_to_vision = True
            self.graph2vis_layer = GraphToVisionReshaper(feat_dim_list=feat_dims,
                                                         current_dataset=self.dataset_name,
                                                         if_feat_avg=if_gr2vis_feat_avg,
                                                         if_additional_reshaper=if_gr2vis_additional,
                                                         batch_size=batch_size,
                                                         )
        elif self.task_type in ['classification', 'action', 'denoising']:
            self.if_graph_to_vision = False
        
        self.timing_info_dict = {}
        timers_in_blocks = ['per_in_gnns', 'graph_samplers', 'graph_pe',
                            'per_out_gnns']
        self.block_timers = {}
        for mod_name in timers_in_blocks:
            self.block_timers[mod_name] = [ProcessingTimer(mode="gpu") for _ in range(num_block_epochs)]
        if self.if_graph_to_vision:
            self.block_timers["graph2vis_layer"] = ProcessingTimer(mode="gpu")

        
    def forward(self, ev_gdata):
        conn_subgraph = ev_gdata
        coords = ev_gdata.orig_coords
        graph_feats_list = []
        unique_coords_list = []

        for i in range(len(self.graph_samplers)):
            timer = self.block_timers['per_in_gnns'][i]
            timer.start()
            conn_subgraph = self.per_in_gnns[i](conn_subgraph) # dim from feat_dims[i-1] to feat_dims[i]
            _, per_in_gnns_gpu_time_i = timer.stop()
            self.timing_info_dict[f"per_in_gnns_{i}"] = per_in_gnns_gpu_time_i

            timer = self.block_timers['graph_samplers'][i]
            timer.start()
            sampled_subgraph, sampled_perm = self.graph_samplers[i](conn_subgraph)
            _, graph_samplers_gpu_time_i = timer.stop()
            self.timing_info_dict[f"graph_samplers_{i}"] = graph_samplers_gpu_time_i
            
            timer = self.block_timers['graph_pe'][i]
            timer.start()
            sampled_subgraph_with_pe = self.graph_pe[i](sampled_subgraph) # with feat_dims[i]
            _, graph_pe_gpu_time_i = timer.stop()
            self.timing_info_dict[f"graph_pe_{i}"] = graph_pe_gpu_time_i

            conn_subgraph, coords, nm_blk_time_dict = self.nodescomps_blocks[i](sampled_subgraph_with_pe)
            for k, v in nm_blk_time_dict.items():
                self.timing_info_dict[f"nodescomps_blocks_{k}_{i}"] = v

            timer = self.block_timers['per_out_gnns'][i]
            timer.start()
            conn_subgraph = self.per_out_gnns[i](conn_subgraph) # dim from feat_dims[i] to feat_dims[i]
            _, per_out_gnns_gpu_time_i = timer.stop()
            self.timing_info_dict[f"per_out_gnns_{i}"] = per_out_gnns_gpu_time_i

            unique_coords_list.append(coords.clone())
            graph_feats_list.append(conn_subgraph.clone())

        if self.if_graph_to_vision: # ['detection', 'tracking']
            reshaped_vision_feats_dict = self.graph2vis_layer(graph_feats_list, unique_coords_list)
            return reshaped_vision_feats_dict, self.timing_info_dict # Dict[int, torch.Tensor]
        
        else: # 'classification'
            return conn_subgraph, self.timing_info_dict # torch.Tensor. The last conn_subgraph

