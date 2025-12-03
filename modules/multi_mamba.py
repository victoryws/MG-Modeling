import torch
import torch.nn as nn
from mamba_ssm import Mamba, Mamba2


class FlexibleMamba(nn.Module):
    def __init__(self, 
                 d_model, 
                 device,
                 d_state, 
                 d_conv, 
                 expand, 
                 mamba_type="mamba2", 
                 if_input_project=False,
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        self.if_input_project = if_input_project
        if if_input_project:
            self.input_proj_layer = nn.Linear(d_model, d_model)
            nn.init.xavier_uniform_(self.input_proj_layer.weight)
            nn.init.zeros_(self.input_proj_layer.bias)
            

        if mamba_type.lower() in ["mamba1", "mamba"]:
            self.mamba = nn.Sequential(
                # nn.LayerNorm(d_model),
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, 
                      d_state=d_state, 
                      d_conv=d_conv, 
                      expand=expand,
                      device=device),
            )
        elif mamba_type.lower() == "mamba2":
            mamba2_headdim = d_model // 2
            mamba2_expand = 4
            assert (d_model * mamba2_expand / mamba2_headdim) % 8 == 0
            self.mamba = nn.Sequential(
                #  nn.LayerNorm(d_model),
                 Mamba2(d_model=d_model, 
                        d_state=d_state, 
                        d_conv=d_conv, 
                        expand=mamba2_expand,
                        headdim=mamba2_headdim,
                        device=device,
                        ).to(device)
            )
        else:
            raise ValueError
        
    def _apply(self, fn):
        module = super()._apply(fn)
        try:
            dev = next(module.mamba.parameters()).device
        except StopIteration:
            dev = None
        if dev is not None and dev.type == 'cuda':
            torch.cuda.set_device(dev)
        return module

    def forward(self, x_in):
        assert x_in.dim() == 3, "input tensor's dim must be 3, [B, N, D]"
        if self.if_input_project:
            x_in = self.input_proj_layer(x_in)
        x_out = self.mamba(x_in)
        return x_out


class MultiMamba(nn.Module):
    def __init__(self, 
                 d_model, 
                 device,
                 d_state=64, 
                 d_conv=4, 
                 expand=2, 
                 mamba_type="mamba", 
                 num_splits=1,
                 reorder_type="one",
                 num_loops=2, 
                 copy_parts_aggr_type="mean", # "max", "min", "sum", "mean", "learned"
                 if_use_residual=True,
                 residual_type='concat', # "simple_sum", "concat", "gate", "proj_sum", "none"
                 
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_loops = num_loops
        self.if_use_residual = if_use_residual
        self.residual_type = residual_type.lower()
        self.reorder_type = reorder_type.lower()

        self.num_splits = num_splits
        self.num_parts = {"three": 3, "six": 6, "one": 1, "two": 2}.get(self.reorder_type, 1)
        self.copy_parts_aggr_type = copy_parts_aggr_type.lower() # "max", "min", "sum", "mean", "learned"
        
        self.mamba_models = nn.ModuleList(
            [FlexibleMamba(d_model=d_model,
                             d_state=d_state,
                             d_conv=d_conv,
                             expand=expand,
                             mamba_type=mamba_type,
                             device=device)
             for _ in range(self.num_parts)]
        )

        valid_res_type = {'simple_sum', 'concat', 'gate', 'proj_sum', 'sigmoid', 'none'}
        if residual_type not in valid_res_type:
            raise ValueError(f"residual_type must be one of {valid_res_type}, got {residual_type!r}")

        if self.if_use_residual:
            if residual_type == 'concat':
                self.concat_layer = nn.Sequential(
                    # nn.RMSNorm(d_model * 2),
                    nn.LayerNorm(d_model * 2),
                    nn.Linear(d_model * 2, d_model * 2),
                    nn.SiLU(),
                    nn.Dropout(0.2),

                    # nn.RMSNorm(d_model * 2),
                    nn.LayerNorm(d_model * 2),
                    nn.Linear(d_model * 2, d_model),
                    nn.SiLU(d_model),
                    nn.Dropout(0.2),
                )
            
            elif residual_type == 'gate':
                self.gate_layer = nn.Sequential(
                    # nn.RMSNorm(d_model * 2),
                    nn.LayerNorm(d_model * 2),
                    nn.Linear(d_model * 2, d_model//4),
                    nn.SiLU(),
                    nn.Dropout(0.2),

                    # nn.RMSNorm(d_model//4),
                    nn.LayerNorm(d_model//4),
                    nn.Linear(d_model//4, 2),
                    nn.Softmax(dim=-1)
                )
            elif residual_type == 'sigmoid':
                # DirectionMamba weighted sum
                self.sigmoid_layer = nn.Sequential(
                    # nn.RMSNorm(d_model),
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 2*d_model),
                    nn.SiLU(),
                    nn.Dropout(0.2),

                    # nn.RMSNorm(d_model*2),
                    nn.LayerNorm(d_model*2),
                    nn.Linear(d_model*2, d_model),
                    nn.Sigmoid(),
                )
            
            elif residual_type == 'proj_sum':
                self.proj_layer = nn.Sequential(
                    # nn.RMSNorm(d_model),
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model*2),
                    nn.SiLU(),
                    nn.Dropout(0.2),

                    # nn.RMSNorm(d_model*2),
                    nn.LayerNorm(d_model*2),
                    nn.Linear(d_model*2, d_model),
                    nn.SiLU(),
                    nn.Dropout(0.2)
                )
        
        self.num_copy = int(self.num_parts/self.num_splits)

        if self.copy_parts_aggr_type == "learned" and self.num_copy>1:
            in_copy_dim = d_model * self.num_copy
            self.learned_layer = nn.Sequential(
                # nn.RMSNorm(in_copy_dim),
                nn.LayerNorm(in_copy_dim),
                nn.Linear(in_copy_dim, d_model*2),
                nn.SiLU(),
                nn.Dropout(0.2),

                # nn.RMSNorm(d_model*2),
                nn.LayerNorm(d_model*2),
                nn.Linear(d_model*2, d_model),
                nn.SiLU(),
                nn.Dropout(0.2),
            )


    def _apply(self, fn):
        super()._apply(fn)
        dev = None
        for p in self.parameters():
            if p.device.type == 'cuda':
                dev = p.device
                break
        if dev is not None:
            self.streams = [torch.cuda.Stream(device=dev) for _ in range(self.num_parts)]
        else:
            self.streams = []
        return self

    def process_residual(self, original, updated):
        rt = self.residual_type
        if rt == 'simple_sum':
            fused = original + updated

        elif rt == 'concat':
            # [B, desired_num, 2d] -> [B, desired_num, d]
            cat = torch.cat([original, updated], dim=-1)
            fused = self.concat_layer(cat)

        elif rt == 'gate':
            cat = torch.cat([original, updated], dim=-1)
            weights = self.gate_layer(cat)    # [B, desired_num, 2]
            w0 = weights[..., 0:1]            # [B, desired_num, 1]
            w1 = weights[..., 1:2]
            fused = original * w0 + updated * w1

        elif rt == 'proj_sum':
            proj_original = self.proj_layer(original)
            fused = proj_original + updated
        
        elif rt == 'sigmoid':
            sig_weights = self.sigmoid_layer(original)  # [B, desired_num, d]
            fused = updated * sig_weights

        else:  # 'none'
            return updated

        return fused

    def reshape_to_orignal_all(self, coords_concat, nodes_concat, comps_concat, indices_concat):
        P, N0, _ = coords_concat.shape  # P = num_parts
        S = self.num_splits
        if S == P:
            merged_coords = coords_concat.reshape(P * N0, 3)
            merged_nodes = nodes_concat.reshape(P * N0, nodes_concat.size(-1))
            merged_comps = comps_concat.reshape(P * N0)
            merged_indices = indices_concat.reshape(P * N0)
            order = torch.argsort(merged_indices, dim=0)
            return merged_coords[order], merged_nodes[order], merged_comps[order]
        else:
            k = P // S
            coords_group = coords_concat.view(k, S, N0, 3)      # [k, S, N0, 3]
            nodes_group = nodes_concat.view(k, S, N0, nodes_concat.size(-1))  # [k, S, N0, D]
            comps_group = comps_concat.view(k, S, N0)            # [k, S, N0]
            indices_group = indices_concat.view(k, S, N0)        # [k, S, N0]
            merged_coords = coords_group.permute(0, 2, 1, 3).reshape(k, S * N0, 3)  # [k, S*N0, 3]
            merged_nodes = nodes_group.permute(0, 2, 1, 3).reshape(k, S * N0, nodes_concat.size(-1))
            merged_comps = comps_group.permute(0, 2, 1).reshape(k, S * N0)
            merged_indices = indices_group.permute(0, 2, 1).reshape(k, S * N0)
            order = torch.argsort(merged_indices, dim=1)
            ret_coords = torch.gather(merged_coords, dim=1, index=order.unsqueeze(-1).expand_as(merged_coords))
            ret_nodes = torch.gather(merged_nodes, dim=1, index=order.unsqueeze(-1).expand_as(merged_nodes))
            ret_comps = torch.gather(merged_comps, dim=1, index=order)
            return ret_coords, ret_nodes, ret_comps

    def process_output_all(self, coords_sorted, nodes_sorted, comps_sorted):
        if coords_sorted.dim() == 2:
            return coords_sorted, nodes_sorted, comps_sorted
        else:
            if self.copy_parts_aggr_type == "mean":
                coords_out = coords_sorted[0]
                nodes_out = nodes_sorted.mean(dim=0)
                comps_out = comps_sorted[0]
            elif self.copy_parts_aggr_type == "max":
                coords_out = coords_sorted[0]
                nodes_out = nodes_sorted.max(dim=0)[0]
                comps_out = comps_sorted[0]
            elif self.copy_parts_aggr_type == "min":
                coords_out = coords_sorted[0]
                nodes_out = nodes_sorted.min(dim=0)[0]
                comps_out = comps_sorted[0]
            elif self.copy_parts_aggr_type == "sum":
                coords_out = coords_sorted[0]
                nodes_out = nodes_sorted.sum(dim=0)
                comps_out = comps_sorted[0]
            elif self.copy_parts_aggr_type == "learned":
                coords_out = coords_sorted[0]
                nodes_sorted_concat = nodes_sorted.permute(1, 2, 0).flatten(1, 2)  # [S*N0, D*k]
                nodes_out = self.learned_layer(nodes_sorted_concat)
                comps_out = comps_sorted[0]
            else:
                raise ValueError
            return coords_out, nodes_out, comps_out

    def forward(self, list_data):
        coords_concat = torch.stack([data[0] for data in list_data], dim=0)    # [num_parts, N0, 3]
        nodes_concat = torch.stack([data[1] for data in list_data], dim=0)     # [num_parts, N0, D]
        comps_concat = torch.stack([data[2] for data in list_data], dim=0)      # [num_parts, N0]
        indices_concat = torch.stack([data[3] for data in list_data], dim=0)    # [num_parts, N0]

        for _ in range(self.num_loops):
            mb_outputs = [None] * self.num_parts
            for i, mb_mdl in enumerate(self.mamba_models):
                with torch.cuda.stream(self.streams[i]):
                    mb_outputs[i] = mb_mdl(nodes_concat[i].unsqueeze(0)).squeeze(0)
            for s in self.streams:
                s.synchronize()
            mb_model_output = torch.stack(mb_outputs, dim=0)  # [num_parts, N0, D]
            if self.if_use_residual:
                nodes_concat = self.process_residual(
                    original=nodes_concat, 
                    updated=mb_model_output)
            else:
                nodes_concat = mb_model_output

        reshaped_coords, reshaped_nodes, reshaped_comps = self.reshape_to_orignal_all(coords_concat, nodes_concat, comps_concat, indices_concat)
        coords_final, nodes_final, comps_final = self.process_output_all(reshaped_coords, reshaped_nodes, reshaped_comps)
        return coords_final, nodes_final, comps_final # [N, 3], [N, D], [N]
