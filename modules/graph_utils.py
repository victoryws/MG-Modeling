import torch
import torch.nn as nn
from torch_geometric.nn import SAGPooling, TopKPooling, ASAPooling
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def get_activation(name):
    """
    Returns the activation function based on the provided name.
    If 'none' is provided, returns nn.Identity().

    Args:
        name (str): Name of the activation function.
            supported activations: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu', 'swish', 'silu', 'softplus', 'none'.

    Returns:
        nn.Module: Corresponding activation function module.

    Raises:
        ValueError: If the activation function name is not recognized.
    """
    name = name.lower()
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        'swish': nn.SiLU,
        'silu': nn.SiLU,
        'softplus': nn.Softplus,
        'none': nn.Identity,
    }
    if name not in activations:
        raise ValueError
    return activations[name]()


# --- DyT 类定义 ---
class DyT(nn.Module):
    """
    DyT (Dynamic Tanh)
    """
    def __init__(self, num_features, alpha_init_value=0.5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init_value), **factory_kwargs))
        self.weight = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        self.num_features = num_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x_transformed = torch.tanh(self.alpha * x)
        return x_transformed * self.weight + self.bias

    def extra_repr(self):
        return f'num_features={self.num_features}, alpha_initial_value={self.alpha.item() if self.alpha.numel() == 1 else self.alpha.shape}'


# --- RMSNorm ---
if hasattr(nn, 'RMSNorm'):
    _RMSNorm = nn.RMSNorm
else:
    class _CustomRMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, device=None, dtype=None):
            super().__init__()
            factory_kwargs = {'device': device, 'dtype': dtype}
            if isinstance(normalized_shape, int):
                self.normalized_shape = (normalized_shape,)
            else:
                self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if self.elementwise_affine:
                self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('weight', None)
            
            self.reset_parameters()

        def reset_parameters(self):
            if self.elementwise_affine:
                nn.init.ones_(self.weight)

        def forward(self, x):
            dims_to_normalize = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))

            dtype = x.dtype
            x_float = x.float()
            denominator = torch.rsqrt(x_float.pow(2).mean(dim=dims_to_normalize, keepdim=True) + self.eps)
            output = (x_float * denominator).to(dtype)

            if self.elementwise_affine:
                output = output * self.weight
            return output

        def extra_repr(self):
            return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

    _RMSNorm = _CustomRMSNorm

# --- get_normalization ---
def get_normalization(name: str, dim: int, **kwargs):
    name = name.lower()
    
    factory_kwargs = {}
    if 'device' in kwargs:
        factory_kwargs['device'] = kwargs.pop('device')
    if 'dtype' in kwargs:
        factory_kwargs['dtype'] = kwargs.pop('dtype')

    if name == 'batch':
        return nn.BatchNorm1d(num_features=dim, **kwargs, **factory_kwargs)
    elif name == 'layer':
        return nn.LayerNorm(normalized_shape=dim, **kwargs, **factory_kwargs)
    elif name == 'rms':
        if hasattr(nn, 'RMSNorm'):
            return nn.RMSNorm(normalized_shape=dim, **kwargs, **factory_kwargs)
        else:
            return _RMSNorm(normalized_shape=dim, **kwargs, **factory_kwargs)
    elif name == 'dyt':
        return DyT(num_features=dim, **kwargs, **factory_kwargs)
    elif name == 'none':
        return nn.Identity(**kwargs)
    elif name == 'instance':
        return nn.InstanceNorm1d(num_features=dim, **kwargs, **factory_kwargs)
    elif name == 'group':
        if dim < 2:
            return nn.Identity()

        num_groups_suggestion = kwargs.pop('num_groups', None)
        
        if num_groups_suggestion is not None:
            if dim % num_groups_suggestion == 0:
                chosen_num_groups = num_groups_suggestion
            else:
                raise ValueError(f"提供的 num_groups={num_groups_suggestion} 不能整除 dim={dim}。")
        else:
            possible_groups = []
            for g in range(min(32, dim), 0, -1):
                if dim % g == 0:
                    possible_groups.append(g)
            
            chosen_num_groups = possible_groups[0]

        return nn.GroupNorm(num_groups=chosen_num_groups, num_channels=dim, **kwargs, **factory_kwargs)
    
    else:
        raise ValueError
# =======================================================================


class GraphSampler:
    def __init__(self, in_dim, ratio=0.5, poolmode='sag'):
        self.ratio = ratio
        self.poolmode = poolmode.lower()
        self.supported_poolmodes = ['topk', 'sag', 'asa', 'random', 'uniform', 'degree', 'random_walk', 'none']

        if self.poolmode not in self.supported_poolmodes:
            raise ValueError(f"Invalid poolmode. Choose from {self.supported_poolmodes}")

        if self.ratio != 1.0 and self.poolmode in ['topk', 'sag', 'asa']:
            self.pool = self._initialize_pool(in_dim)
        else:
            self.pool = None

    def _initialize_pool(self, in_dim):
        if self.poolmode == 'topk':
            return TopKPooling(in_dim, ratio=self.ratio)
        elif self.poolmode == 'sag':
            return SAGPooling(in_dim, ratio=self.ratio)
        elif self.poolmode == 'asa':
            return ASAPooling(in_dim, ratio=self.ratio)
        else:
            return None

    def sample_graph(self, graph):
        device = graph.x.device
        original_num_nodes = graph.num_nodes

        if self.ratio == 1.0 or self.poolmode == 'none':
            perm = torch.arange(original_num_nodes, device=device)
            return graph, perm

        if self.pool is not None:
            self.pool = self.pool.to(device)
            x, edge_index, batch = graph.x, graph.edge_index, graph.batch
            if self.poolmode in ['topk', 'sag']:
                x, edge_index, _, batch, perm, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
            elif self.poolmode == 'asa':
                x, edge_index, edge_weight, batch, perm = self.pool(x=x, edge_index=edge_index, batch=batch)
            sampled_graph = Data(x=x, edge_index=edge_index, batch=batch)

            self._reindex_attributes(graph, sampled_graph, perm, original_num_nodes)

            return sampled_graph, perm
        else:
            if self.poolmode in ['random', 'uniform']:
                return self._random_or_uniform_sample(graph, original_num_nodes)
            elif self.poolmode == 'degree':
                return self._degree_based_sample(graph, original_num_nodes)
            elif self.poolmode == 'random_walk':
                return self._random_walk_sample(graph, original_num_nodes)
            else:
                raise ValueError(f"Unsupported poolmode: {self.poolmode}")

    def _random_or_uniform_sample(self, graph, original_num_nodes):
        device = graph.x.device
        num_nodes_to_keep = max(int(original_num_nodes * self.ratio), 1)

        if self.poolmode == 'random':
            perm = torch.randperm(original_num_nodes, device=device)[:num_nodes_to_keep]
        elif self.poolmode == 'uniform':
            step = max(int(original_num_nodes / num_nodes_to_keep), 1)
            perm = torch.arange(0, original_num_nodes, step=step, device=device)[:num_nodes_to_keep]
        else:
            raise ValueError(f"Unsupported poolmode in _random_or_uniform_sample: {self.poolmode}")

        new_edge_index, _ = subgraph(perm, graph.edge_index, relabel_nodes=True, num_nodes=original_num_nodes)
        new_x = graph.x[perm]
        new_batch = graph.batch[perm] if hasattr(graph, 'batch') else None

        sampled_graph = Data(x=new_x, edge_index=new_edge_index, batch=new_batch)

        self._reindex_attributes(graph, sampled_graph, perm, original_num_nodes)

        return sampled_graph, perm

    def _degree_based_sample(self, graph, original_num_nodes):
        device = graph.x.device
        num_nodes_to_keep = max(int(original_num_nodes * self.ratio), 1)

        degrees = torch.bincount(graph.edge_index[0], minlength=original_num_nodes).float()

        perm = degrees.topk(num_nodes_to_keep, largest=True, sorted=False).indices

        new_edge_index, _ = subgraph(perm, graph.edge_index, relabel_nodes=True, num_nodes=original_num_nodes)
        new_x = graph.x[perm]
        new_batch = graph.batch[perm] if hasattr(graph, 'batch') else None

        sampled_graph = Data(x=new_x, edge_index=new_edge_index, batch=new_batch)

        self._reindex_attributes(graph, sampled_graph, perm, original_num_nodes)

        return sampled_graph, perm

    def _random_walk_sample(self, graph, original_num_nodes, walk_length=2, num_walks=10):
        device = graph.x.device
        num_nodes_to_keep = max(int(original_num_nodes * self.ratio), 1)

        walks = []
        for _ in range(num_walks):
            start_node = torch.randint(0, original_num_nodes, (1,), device=device).item()
            walk = [start_node]
            current_node = start_node
            for _ in range(walk_length - 1):
                neighbors = graph.edge_index[1][graph.edge_index[0] == current_node]
                if len(neighbors) == 0:
                    break
                current_node = neighbors[torch.randint(0, len(neighbors), (1,), device=device)].item()
                walk.append(current_node)
            walks.extend(walk)

        walk_nodes = torch.tensor(walks, device=device).unique()

        perm = walk_nodes[:num_nodes_to_keep]

        if perm.numel() < num_nodes_to_keep:
            remaining = num_nodes_to_keep - perm.numel()
            additional = torch.randperm(original_num_nodes, device=device)[:remaining]
            perm = torch.cat([perm, additional], dim=0)

        new_edge_index, _ = subgraph(perm, graph.edge_index, relabel_nodes=True, num_nodes=original_num_nodes)
        new_x = graph.x[perm]
        new_batch = graph.batch[perm] if hasattr(graph, 'batch') else None

        sampled_graph = Data(x=new_x, edge_index=new_edge_index, batch=new_batch)

        self._reindex_attributes(graph, sampled_graph, perm, original_num_nodes)

        return sampled_graph, perm

    def _reindex_attributes(self, original_graph, sampled_graph, perm, original_num_nodes):
        for key in original_graph.keys:
            if key in ['x', 'edge_index', 'batch', 'ptr', 'pos']:
                continue
            attr = original_graph[key]
            if isinstance(attr, torch.Tensor) and attr.size(0) == original_num_nodes:
                sampled_graph[key] = attr[perm]

    def __call__(self, graph):
        return self.sample_graph(graph)




class EventDataProcessor:
    def __init__(self,
                 camera_x: int,
                 camera_y: int,
                 if_normalization: bool = True,
                 if_standardization: bool = False):
        if if_normalization and if_standardization:
            raise ValueError
        elif not if_normalization and not if_standardization:
            raise ValueError
        
        self.camera_x = camera_x
        self.camera_y = camera_y
        self.if_normalization = if_normalization
        self.if_standardization = if_standardization

        self.mean_data = None
        self.std_data = None

        if self.if_normalization:
            self.inv_camera_x = 1.0 / self.camera_x
            self.inv_camera_y = 1.0 / self.camera_y

    def _check_data_validity(self, data: torch.Tensor):
        if data.shape[1] < 3:
            raise ValueError

    def _compute_mean_and_std(self, data: torch.Tensor):
        if self.if_standardization:
            self.mean_data = data.mean(dim=0)
            self.std_data = data.std(dim=0)
            self.std_data[self.std_data == 0] = 1.0

    def _norm_and_standard(self, data: torch.Tensor) -> torch.Tensor:
        if self.if_normalization:
            data[:, 0].mul_(self.inv_camera_x)
            data[:, 1].mul_(self.inv_camera_y)
            
            t_min = data[:, 2].min(dim=0, keepdim=True).values
            t_max = data[:, 2].max(dim=0, keepdim=True).values
            t_range = t_max - t_min
            non_zero = t_range != 0
            data[:, 2] = torch.where(non_zero, (data[:, 2] - t_min) / t_range, torch.zeros_like(data[:, 2]))
        
        if self.if_standardization and self.mean_data is not None and self.std_data is not None:
            data -= self.mean_data.to(data.device)
            data /= self.std_data.to(data.device)
        
        return data

    def transform(self, input_data):
        if isinstance(input_data, torch.Tensor):
            data = input_data
            self._check_data_validity(data)
            if self.if_standardization:
                self._compute_mean_and_std(data)
            self._norm_and_standard(data)
            return data

        elif hasattr(input_data, 'x') and isinstance(input_data.x, torch.Tensor):
            data = input_data.x
            self._check_data_validity(data)
            if self.if_standardization:
                self._compute_mean_and_std(data)
            self._norm_and_standard(data)
            return input_data

        else:
            raise TypeError

    def __call__(self, input_data):
        return self.transform(input_data)

