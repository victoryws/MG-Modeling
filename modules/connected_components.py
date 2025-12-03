import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix


class CustomLargestConnectedComponents(BaseTransform):
    def __init__(self, 
                 num_components: int = None, 
                 min_nodes_cnt_in_subgraph: int = 1, 
                 min_mean_ratio: float = 0.0, 
                 connection: str = 'weak'):
        assert connection in ['strong', 'weak'], 'Unknown connection type'
        if num_components is not None:
            assert num_components > 0, 'num_components 必须是正整数或 None'
            
        self.num_components = num_components
        self.min_nodes_cnt_in_subgraph = min_nodes_cnt_in_subgraph
        self.min_mean_ratio = min_mean_ratio
        self.connection = connection

    def forward(self, data: Data) -> Data:
        # --- stage 1 ---
        num_nodes = self._get_num_nodes(data)
        if num_nodes == 0:
             return self._empty_data(data)

        edge_index_np = data.edge_index.numpy()
        
        adj = sp.coo_matrix(
            (np.ones(edge_index_np.shape[1]), (edge_index_np[0], edge_index_np[1])),
            shape=(num_nodes, num_nodes)
        )
        
        num_total, component_labels = sp.csgraph.connected_components(
            adj, connection=self.connection
        )
        unique_labels, counts = np.unique(component_labels, return_counts=True)
        
        if len(counts) == 0:
            return self._empty_data(data)

        # --- stage 2 ---
        final_valid_labels = self._step2_filter_components(unique_labels, counts)
        
        if len(final_valid_labels) == 0:
            return self._empty_data(data)

        # --- stage 3 ---
        sub_data = self._step3_extract_subgraph(data, component_labels, final_valid_labels, num_total)

        # --- stage 4 ---
        sub_data = self._step4_pad_nodes(sub_data)
        
        return sub_data

    # =========================================================================
    #                               private methods
    # =========================================================================

    def _get_num_nodes(self, data: Data) -> int:
        if hasattr(data, 'x') and data.x is not None:
            return data.x.size(0)
        elif hasattr(data, 'pos') and data.pos is not None:
            return data.pos.size(0)
        elif hasattr(data, 'adj') and data.adj is not None:
             return data.adj.size(0)
        elif hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
            return int(data.edge_index.max()) + 1
        else:
            return 0
    
    def _empty_data(self, data):
        x_dim = data.x.size(1) if data.x is not None else 0
        return Data(edge_index=torch.empty((2, 0), dtype=torch.long, device=data.edge_index.device), 
                    x=torch.zeros(0, x_dim, device=data.edge_index.device) if data.x is not None else None)

    def _step1_compute_components(self, data: Data):
        num_nodes = self._get_num_nodes(data)
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
        num_total, component_labels = sp.csgraph.connected_components(
            adj, connection=self.connection
        )
        unique_labels, counts = np.unique(component_labels, return_counts=True)
        return num_total, component_labels, unique_labels, counts

    def _step2_filter_components(self, unique_labels: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if self.min_nodes_cnt_in_subgraph > 1:
            keep_mask = (counts >= self.min_nodes_cnt_in_subgraph)
        else:
            keep_mask = np.ones(len(counts), dtype=bool)

        if not np.any(keep_mask): return np.array([])

        if self.min_mean_ratio > 0:
            valid_counts_after_abs = counts[keep_mask]
            if len(valid_counts_after_abs) > 0:
                mean_size = np.mean(valid_counts_after_abs)
                adaptive_threshold = mean_size * self.min_mean_ratio
                keep_mask &= (counts >= adaptive_threshold)

        valid_labels = unique_labels[keep_mask]
        valid_counts = counts[keep_mask]

        if len(valid_counts) == 0: return np.array([])

        if self.num_components is not None and len(valid_counts) > self.num_components:
            top_k_indices = np.argsort(valid_counts)[-self.num_components:]
            final_labels = valid_labels[top_k_indices]
        else:
            final_labels = valid_labels

        return final_labels

    def _step3_extract_subgraph(self, data: Data, component_labels: np.ndarray, 
                                final_valid_labels: np.ndarray, num_total_components: int) -> Data:
        final_valid_labels = np.sort(final_valid_labels)
        
        new_label_mapping = np.full(num_total_components, -1, dtype=int)
        new_label_mapping[final_valid_labels] = np.arange(len(final_valid_labels))

        new_node_labels = new_label_mapping[component_labels] 
        
        node_mask_np = (new_node_labels != -1)
        
        node_mask_tensor = torch.from_numpy(node_mask_np)
        
        sub_data = data.subgraph(node_mask_tensor)

        valid_labels_cpu = new_node_labels[node_mask_np]
        sub_data.component_labels = torch.from_numpy(valid_labels_cpu)

        return sub_data

    def _step4_pad_nodes(self, graph_data: Data) -> Data:
        n = self._get_num_nodes(graph_data)
        if n == 0: return graph_data

        remainder = n % 3
        if remainder != 0:
            num_pad = 3 - remainder
            keys = list(graph_data.keys())
            
            for key in keys:
                if key == "edge_index": continue
                attr = graph_data[key]
                
                if isinstance(attr, torch.Tensor) and attr.dim() >= 1 and attr.size(0) == n:
                    
                    last_node_attr = attr[-1:] 
                    
                    padding_block = last_node_attr.expand(num_pad, *attr.shape[1:])
                    
                    graph_data[key] = torch.cat([attr, padding_block], dim=0)

        return graph_data


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_components={self.num_components}, '
                f'min_mean_ratio={self.min_mean_ratio}, '
                f'min_nodes={self.min_nodes_cnt_in_subgraph})')

