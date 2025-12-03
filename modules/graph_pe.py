import torch
import torch.nn as nn
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


class CoordinatesPE(nn.Module):
    def __init__(self, in_dim=3, out_dim=8):
        super().__init__()
        hidden_dims = get_power_of_2_hidden_dims(in_dim, out_dim, num_layers=3)
        self.coords_pe_layer = nn.Sequential(
            nn.RMSNorm(in_dim),
            nn.Linear(in_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(0.2),

            # nn.RMSNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.RMSNorm(hidden_dims[1]),
            nn.Linear(hidden_dims[1], out_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.coords_pe_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, coords):
        coords_pe = self.coords_pe_layer(coords)
        return coords_pe



class ComponentsPE(nn.Module):
    def __init__(self, num_components=2048, out_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_components, embedding_dim=out_dim)
        self.norm = nn.RMSNorm(out_dim)
        self.dropout = nn.Dropout(0.2)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, component_labels):
        comps_pe = self.embedding(component_labels)
        comps_pe = self.norm(comps_pe)
        comps_pe = self.dropout(comps_pe)
        return comps_pe


class SinusoidalCoordsPE(nn.Module):
    def __init__(self, coord_dim=3, out_dim=8, n_freq_bands=4, include_original=False):
        """
        Generates sinusoidal positional embeddings for n-dimensional coordinates.
        Args:
            coord_dim (int): Dimensionality of the input coordinates (e.g., 3 for 3D).
            out_dim (int): Desired output dimensionality of the PE.
            n_freq_bands (int): Number of frequency bands for sinusoidal encoding per coordinate dimension.
                                The raw sinusoidal features per coordinate will be 2 * n_freq_bands.
            include_original (bool): Whether to include original coordinates in the output before projection.
        """
        super().__init__()
        self.coord_dim = coord_dim
        self.out_dim = out_dim
        self.n_freq_bands = n_freq_bands
        self.include_original = include_original

        self.raw_sincos_dim_per_coord = 2 * self.n_freq_bands
        total_raw_dim = self.coord_dim * self.raw_sincos_dim_per_coord
        if self.include_original:
            total_raw_dim += self.coord_dim
        
        self.projector = nn.Linear(total_raw_dim, out_dim) if total_raw_dim != out_dim else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        if isinstance(self.projector, nn.Linear):
            nn.init.xavier_uniform_(self.projector.weight)
            if self.projector.bias is not None:
                nn.init.zeros_(self.projector.bias)

    def forward(self, coords):
        """
        coords: Tensor of shape [n_nodes, coord_dim]
        """
        if coords.shape[1] != self.coord_dim:
            raise ValueError(f"Input coords dim {coords.shape[1]} != self.coord_dim {self.coord_dim}")

        embedded_coords = []
        if self.include_original:
            embedded_coords.append(coords)

        for i in range(self.n_freq_bands):
            freq = (2.0**i) * math.pi
            embedded_coords.append(torch.sin(coords * freq))
            embedded_coords.append(torch.cos(coords * freq))
        
        raw_pe = torch.cat(embedded_coords, dim=-1)
        projected_pe = self.projector(raw_pe)
        return projected_pe


class LaplacianPE(nn.Module):
    def __init__(self, num_eigenvectors, out_dim):
        """
        Projects pre-computed Laplacian eigenvectors to the desired output dimension.
        Args:
            num_eigenvectors (int): The number of eigenvectors provided as input (k).
            out_dim (int): Desired output dimensionality.
        """
        super().__init__()
        self.projector = nn.Linear(num_eigenvectors, out_dim)
        self.norm = nn.LayerNorm(out_dim) # Optional: normalize after projection
        self.act = nn.SiLU() # Optional: activation after projection
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projector.weight)
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)
            
    def forward(self, lap_eigvecs):
        """
        lap_eigvecs: Tensor of shape [n_nodes, num_eigenvectors]
        """
        pe = self.projector(lap_eigvecs)
        pe = self.act(pe)
        pe = self.norm(pe)
        return pe


class GraphPositionalEncoding(nn.Module):
    def __init__(self, out_dim=64, num_components=128, if_use_pe=True,
                 # --------------------------------
                 pe_type='learned', 
                 # Args for 'learned_coords_comps'
                 coord_in_dim=3, 
                 pe_dim=16, 
                 # Args for 'sincos_coords'
                 sincos_coord_dim=3, sincos_n_freq_bands=4, sincos_include_original=False,
                 # Args for 'laplacian'
                 lap_num_eigenvectors=10):
        super().__init__()
        self.if_use_pe = if_use_pe
        self.pe_type = pe_type
        self.out_dim = out_dim

        if not self.if_use_pe:
            return

        if self.pe_type == 'learned':
            self.coords_pe_layer = CoordinatesPE(in_dim=coord_in_dim, out_dim=pe_dim)
            self.comps_pe_layer = ComponentsPE(num_components=num_components, out_dim=pe_dim)
            self.coords_comps_pe_layer = nn.Sequential(
                nn.RMSNorm(pe_dim*2),
                nn.Linear(pe_dim*2, pe_dim*2),
                nn.SiLU(),
                nn.Dropout(0.2),

                # nn.RMSNorm(hidden_dims[0]),
                nn.Linear(pe_dim*2, pe_dim),
                nn.SiLU(),
                nn.Dropout(0.2),

                nn.RMSNorm(pe_dim),
                nn.Linear(pe_dim, pe_dim),
                nn.SiLU(),
                nn.Dropout(0.2),
            )

            self.x_coords_comps_pe_layer = nn.Sequential(
                nn.RMSNorm(pe_dim+out_dim),
                nn.Linear(pe_dim+out_dim, pe_dim+out_dim),
                nn.SiLU(),
                nn.Dropout(0.2),

                nn.RMSNorm(pe_dim+out_dim),
                nn.Linear(pe_dim+out_dim, out_dim),
                nn.SiLU(),
                nn.Dropout(0.2),
            )
        elif self.pe_type == 'sincos':
            self.sincos_coords_pe_layer = SinusoidalCoordsPE(
                coord_dim=sincos_coord_dim, 
                out_dim=out_dim, 
                n_freq_bands=sincos_n_freq_bands,
                include_original=sincos_include_original
            )
        elif self.pe_type == 'laplacian':
            self.laplacian_pe_layer = LaplacianPE(
                num_eigenvectors=lap_num_eigenvectors,
                out_dim=out_dim
            )
        elif self.pe_type == 'none':
            pass
        else:
            raise ValueError(f"Unsupported pe_type: {self.pe_type}")

    def forward(self, graph, lap_eigvecs=None):
        """
        Args:
            graph: A graph object. Expected to have:
                   - graph.orig_coords (for 'learned_coords_comps', 'sincos_coords')
                   - graph.component_labels (for 'learned_coords_comps')
                   - graph.num_nodes (for returning zero tensor if needed)
                   - graph.x (optional, for device and shape reference if returning zeros)
            lap_eigvecs (Tensor, optional): Pre-computed Laplacian eigenvectors of shape [n_nodes, k].
                                           Required if pe_type is 'laplacian'.
        """
        if not self.if_use_pe or self.pe_type == 'none':
            # Determine device and num_nodes
            if hasattr(graph, 'x') and graph.x is not None:
                device = graph.x.device
                # num_nodes = graph.x.shape[0]
            elif hasattr(graph, 'orig_coords') and graph.orig_coords is not None:
                device = graph.orig_coords.device
                # num_nodes = graph.orig_coords.shape[0]
            elif hasattr(graph, 'num_nodes'):
                device = 'cpu' # Default device if not inferable
                if lap_eigvecs is not None:
                    device = lap_eigvecs.device

            else:
                raise ValueError
            return torch.zeros_like(graph.x, device=device) # Return zero tensor if PE is not used

        x_feats = graph.x
        nodes_pe = None
        if self.pe_type == 'learned':
            coords, component_labels = graph.orig_coords, graph.component_labels
            if coords.shape[0] != component_labels.shape[0]:
                 raise AssertionError("Coords and component_labels must have the same number of nodes.")
            
            coords_pe = self.coords_pe_layer(coords)
            comps_pe = self.comps_pe_layer(component_labels)
            
            coords_comps_pe = self.coords_comps_pe_layer(torch.cat([coords_pe, comps_pe], dim=-1))

            nodes_pe = self.x_coords_comps_pe_layer(torch.cat([coords_comps_pe, x_feats], dim=-1))
            graph.x = nodes_pe

        elif self.pe_type == 'sincos':
            coords = graph.orig_coords
            nodes_pe = self.sincos_coords_pe_layer(coords)
            graph.x = nodes_pe + x_feats

        elif self.pe_type == 'laplacian':
            if lap_eigvecs is None:
                raise ValueError
            if lap_eigvecs.shape[1] != self.laplacian_pe_layer.projector.in_features:
                raise ValueError
            nodes_pe = self.laplacian_pe_layer(lap_eigvecs)
            graph.x = nodes_pe + x_feats  # Update graph.x if needed

        if nodes_pe is None: # Should not happen if logic is correct
             raise RuntimeError
             
        return graph

