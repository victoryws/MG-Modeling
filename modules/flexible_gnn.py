import torch
import torch.nn as nn
from eg_mamba.modules.graph_utils import get_activation, get_normalization
from torch_geometric.nn import GCNConv, GATConv,GATv2Conv, EdgeConv, SAGEConv, GINConv, GraphConv, ChebConv, SGConv, APPNP, ARMAConv, GatedGraphConv, GMMConv, SignedConv, DNAConv, GENConv


class FlexibleGNN(nn.Module):
    """
    A flexible GNN module that can dynamically select different GNN layers, number of layers,
    activation functions, and normalization layers based on input parameters.
    """
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dims_list=None, 
                 if_use_gnn=True,
                 num_layers=1, 
                 gnn_type='GCN', 
                 gat_heads=2, # Default number of heads for GAT/GATv2
                 activation='none', 
                 normalization='rms', 
                 if_use_dropout=False, 
                 dropout=0.1, 
                 if_use_skip_connection=False):
        super().__init__()
        
        # Store parameters
        self.if_use_gnn = if_use_gnn
        if not self.if_use_gnn:
            # If not using GNN, no need to initialize other components
            return
        
        # Parameter validation
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.upper()
        self.activation = activation
        self.normalization = normalization
        self.if_use_dropout = if_use_dropout
        self.dropout = nn.Dropout(p=dropout) if if_use_dropout else nn.Identity()
        self.if_use_skip_connection = if_use_skip_connection
        self.gat_heads = gat_heads  # Default number of heads for GAT/GATv2
        
        # Define supported GNN layers
        self.supported_gnn = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'GATV2': GATv2Conv,
            'SAGE': SAGEConv,
            'GIN': GINConv,
            'GRAPHCONV': GraphConv,
            'CHEBCONV': ChebConv,
            'SGCONV': SGConv,
            'EDGE': EdgeConv,
        }
        
        if self.gnn_type not in self.supported_gnn:
            raise ValueError(f"GNN type '{gnn_type}' is not supported. Supported types: {list(self.supported_gnn.keys())}")
        
        self.ConvLayer = self.supported_gnn[self.gnn_type]
        
        # Automatically compute hidden dimensions based on num_layers
        self.hidden_dims = self.compute_hidden_dims(hidden_dims_list)
        
        # Build GNN layers, activations, and normalization layers
        self.convs, self.acts, self.norms, self.skip_projections = self.build_layers()

        self.apply(init_weights)
        
    def compute_hidden_dims(self, hidden_dims_list=None):
        target_count = self.num_layers - 1
        
        if hidden_dims_list is None:
            if target_count <= 0:
                self.hidden_dims = []
            else:
                import math
                
                start_log = math.log2(max(self.input_dim, 1))
                end_log = math.log2(max(self.output_dim, 1))
                
                step = (end_log - start_log) / (target_count + 1)
                
                generated_dims = []
                for i in range(1, target_count + 1):
                    current_log = start_log + step * i
                    dim_val = 2 ** current_log
                    
                    power_of_2 = 2 ** round(math.log2(dim_val))
                    generated_dims.append(int(power_of_2))
                
                self.hidden_dims = generated_dims
        else:
            assert len(hidden_dims_list) == target_count, \
                f"hidden_dims_list length ({len(hidden_dims_list)}) must be num_layers - 1 ({target_count})"
            self.hidden_dims = hidden_dims_list
            
        return self.hidden_dims
    
    def build_layers(self):
        """
        Build GNN layers, activation functions, normalization layers, and skip_projections.

        Returns:
            Tuple[nn.ModuleList, nn.ModuleList, nn.ModuleList]: GNN layers, activation functions, normalization layers.
        """
        convs = nn.ModuleList()
        acts = nn.ModuleList()
        norms = nn.ModuleList()
        skip_projections = nn.ModuleList()
        
        for layer_idx in range(self.num_layers):
            in_dim = self.input_dim if layer_idx == 0 else self.hidden_dims[layer_idx - 1]
            out_dim = self.output_dim if layer_idx == self.num_layers - 1 else self.hidden_dims[layer_idx]
            
            # Special handling for certain GNN layers
            if self.gnn_type == 'GAT':
                conv = self.ConvLayer(in_channels=in_dim, out_channels=out_dim, heads=self.gat_heads, concat=False)
            elif self.gnn_type == 'GATV2':
                conv = self.ConvLayer(in_channels=in_dim, out_channels=out_dim, heads=self.gat_heads, concat=False)
            elif self.gnn_type == 'SAGE':
                conv = self.ConvLayer(in_channels=in_dim, out_channels=out_dim, aggr='max')
            elif self.gnn_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, in_dim*2),
                    nn.SiLU(),
                    nn.Dropout(0.2),

                    # nn.RMSNorm(in_dim*2),
                    nn.LayerNorm(in_dim*2),
                    nn.Linear(in_dim*2, out_dim),
                    nn.SiLU(),
                    nn.Dropout(0.2),
                )
                conv = self.ConvLayer(mlp)

            elif self.gnn_type == 'EDGE':
                # EdgeConv requires an MLP, similar to GIN
                # MLP input dim is 2 * in_dim (x_i, x_j - x_i)
                mlp = nn.Sequential(
                    nn.Linear(2 * in_dim, in_dim*4),
                    nn.SiLU(),
                    nn.Dropout(0.2),

                    # nn.RMSNorm(in_dim*4),
                    nn.LayerNorm(in_dim*4),
                    nn.Linear(in_dim*4, out_dim),
                    nn.SiLU(),
                    nn.Dropout(0.2),
                )
                conv = self.ConvLayer(mlp)

            elif self.gnn_type == 'CHEBCONV':
                conv = self.ConvLayer(in_channels=in_dim, out_channels=out_dim, K=3)
            elif self.gnn_type == 'SGCONV':
                conv = self.ConvLayer(in_channels=in_dim, out_channels=out_dim, K=2)
            else:
                conv = self.ConvLayer(in_channels=in_dim, out_channels=out_dim)
            
            convs.append(conv)
            
            acts.append(get_activation(self.activation))
            norms.append(get_normalization(self.normalization, in_dim))
        
            if self.if_use_skip_connection:
                if in_dim != out_dim:
                    projection = nn.Linear(in_dim, out_dim)
                else:
                    projection = nn.Identity()
                skip_projections.append(projection)
        
        return convs, acts, norms, skip_projections
    
    def forward(self, graph):
        """
        Forward pass of the GNN module.

        Args:
            graph (torch_geometric.data.Data): Input graph data.

        Returns:
            torch_geometric.data.Data: Output graph data.
        """
        if not self.if_use_gnn:
            return graph  # If not using GNN, return the input graph unchanged
        
        x, edge_index = graph.x, graph.edge_index
        
        for layer in range(self.num_layers):
            x_residual = x
            x = self.norms[layer](x)
            x = self.convs[layer](x, edge_index)
            x = self.acts[layer](x)
            if self.if_use_dropout:
                x = self.dropout(x)
            
            if self.if_use_skip_connection:
                projection = self.skip_projections[layer]
                skip_projected = projection(x_residual) 
                x = x + skip_projected
        
        graph.x = x
        return graph
    


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, GCNConv):
        if hasattr(module, 'lin') and hasattr(module.lin, 'weight'):
            nn.init.xavier_uniform_(module.lin.weight)
            if hasattr(module.lin, 'bias') and module.lin.bias is not None:
                nn.init.zeros_(module.lin.bias)

        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, GATConv):
        if hasattr(module, 'lin_src') and hasattr(module.lin_src, 'weight'):
            nn.init.xavier_uniform_(module.lin_src.weight)
            if hasattr(module.lin_src, 'bias') and module.lin_src.bias is not None:
                nn.init.zeros_(module.lin_src.bias)
        if hasattr(module, 'att_src'):
            nn.init.xavier_uniform_(module.att_src)
        if hasattr(module, 'att_dst'):
            nn.init.xavier_uniform_(module.att_dst)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, GATv2Conv):
        if hasattr(module, 'lin_l') and hasattr(module.lin_l, 'weight'):
            nn.init.xavier_uniform_(module.lin_l.weight)
            if hasattr(module.lin_l, 'bias') and module.lin_l.bias is not None:
                nn.init.zeros_(module.lin_l.bias)
        if hasattr(module, 'lin_r') and hasattr(module.lin_r, 'weight'):
             nn.init.xavier_uniform_(module.lin_r.weight)
             if hasattr(module.lin_r, 'bias') and module.lin_r.bias is not None:
                 nn.init.zeros_(module.lin_r.bias)
        if hasattr(module, 'att'):
            nn.init.xavier_uniform_(module.att)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, SAGEConv):
        if hasattr(module, 'lin_l') and hasattr(module.lin_l, 'weight'):
            nn.init.xavier_uniform_(module.lin_l.weight)
            if hasattr(module.lin_l, 'bias') and module.lin_l.bias is not None:
                nn.init.zeros_(module.lin_l.bias)

        if hasattr(module, 'lin_r') and hasattr(module.lin_r, 'weight'):
            nn.init.xavier_uniform_(module.lin_r.weight)
            if hasattr(module.lin_r, 'bias') and module.lin_r.bias is not None:
                nn.init.zeros_(module.lin_r.bias)
        return

    if isinstance(module, (GINConv, EdgeConv)):
        if hasattr(module, 'nn'):
            for layer in module.nn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        return

    if isinstance(module, GraphConv):
        if hasattr(module, 'lin_rel') and hasattr(module.lin_rel, 'weight'):
            nn.init.xavier_uniform_(module.lin_rel.weight)
            if hasattr(module.lin_rel, 'bias') and module.lin_rel.bias is not None:
                nn.init.zeros_(module.lin_rel.bias)
        if hasattr(module, 'lin_root') and hasattr(module.lin_root, 'weight'):
            nn.init.xavier_uniform_(module.lin_root.weight)
        return

    if isinstance(module, ChebConv):
        if hasattr(module, 'lins'):
            for lin in module.lins:
                if hasattr(lin, 'weight'):
                    nn.init.xavier_uniform_(lin.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, SGConv):
        if hasattr(module, 'lin') and hasattr(module.lin, 'weight'):
            nn.init.xavier_uniform_(module.lin.weight)
            if hasattr(module, 'bias') and module.lin.bias is not None:
                nn.init.zeros_(module.lin.bias)
        return
