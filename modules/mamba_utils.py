import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_min, scatter_max, scatter_add


def compute_aggregation_batch(coords, nodes, component_labels,
                              coords_operation='xyt_mean', nodes_operation='sum'):
    aggr_comps = torch.unique(component_labels)

    aggr_coords = None
    perm = None

    if coords_operation == 'xyt_mean':
        aggr_coords = scatter_mean(coords, component_labels, dim=0)
        perm = None
        
    elif coords_operation == 't_min':
        _, indices = scatter_min(coords[:, 2], component_labels, dim=0)
        aggr_coords = coords.index_select(0, indices)
        perm = indices
        
    elif coords_operation == 't_max':
        _, indices = scatter_max(coords[:, 2], component_labels, dim=0)
        aggr_coords = coords.index_select(0, indices)
        perm = indices
        
    elif coords_operation == 'xytsum_min':
        sum_val = coords.sum(dim=1)
        _, indices = scatter_min(sum_val, component_labels, dim=0)
        aggr_coords = coords.index_select(0, indices)
        perm = indices
        
    elif coords_operation == 'xytsum_max':
        sum_val = coords.sum(dim=1)
        _, indices = scatter_max(sum_val, component_labels, dim=0)
        aggr_coords = coords.index_select(0, indices)
        perm = indices
        
    else:
        raise ValueError
    
    aggr_nodes = None
    
    if nodes_operation == 'mean':
        aggr_nodes = scatter_mean(nodes, component_labels, dim=0)
        
    elif nodes_operation == 'sum':
        aggr_nodes = scatter_add(nodes, component_labels, dim=0)
        
    elif nodes_operation == 'max':
        aggr_nodes, _ = scatter_max(nodes, component_labels, dim=0)
        
    elif nodes_operation == 'min':
        aggr_nodes, _ = scatter_min(nodes, component_labels, dim=0)
        
    else:
        raise ValueError

    return aggr_coords, aggr_nodes, aggr_comps, perm




def split_and_reorder_coords_nodes_comps(
    coords, 
    nodes, 
    component_labels, 
    split_parts, 
    reorder_type="one"
):
    num_nodes = nodes.size(0)
    
    orig_idx = torch.arange(num_nodes, device=coords.device)
    
    if split_parts == 1:
        assert reorder_type in {"one", "two", "three", "six"}, \
            "split_parts=1, reorder_type must be 'one','two','three' or 'six'."
    elif split_parts == 2:
        assert reorder_type in {"two", "six"}, \
            "split_parts=2, reorder_type must be 'two' or 'six'."
    elif split_parts == 3:
        assert num_nodes % 3 == 0, f"num_nodes ({num_nodes}) must be a multiple of 3."
        assert reorder_type in {"three", "six"}, \
            "split_parts=3, reorder_type must be 'three' or 'six'."
    elif split_parts == 6:
        assert num_nodes % 6 == 0, f"num_nodes ({num_nodes}) must be a multiple of 6."
        assert reorder_type == "six", \
            "split_parts=6, reorder_type must be 'six'."
    else:
        raise ValueError("split_parts must be 1, 2, 3 or 6.")

    def composite_sort(in_coords, in_nodes, in_comps, in_idx, coord_dim, descending=False):
        col = in_coords[:, coord_dim]
        col_min = col.min()
        col_max = col.max()
        factor = (col_max - col_min).item() + 10.0
        if factor <= 0:
            factor = 10.0
        composite_key = in_comps.to(in_coords.dtype) * factor + in_coords[:, coord_dim]
        sorted_idx = torch.sort(composite_key, descending=descending, stable=True).indices
        return (in_coords.index_select(0, sorted_idx),
                in_nodes.index_select(0, sorted_idx),
                in_comps.index_select(0, sorted_idx),
                in_idx.index_select(0, sorted_idx))
    
    # ========== split_parts == 1 ==========
    if split_parts == 1:
        if reorder_type == "one":
            return [composite_sort(coords, nodes, component_labels, orig_idx, coord_dim=2, descending=False)]
        elif reorder_type == "two":
            asc = composite_sort(coords, nodes, component_labels, orig_idx, coord_dim=2, descending=False)
            desc = composite_sort(coords, nodes, component_labels, orig_idx, coord_dim=2, descending=True)
            return [asc, desc]
        elif reorder_type == "three":
            results = []
            for dim in [0, 1, 2]:
                results.append(composite_sort(coords, nodes, component_labels, orig_idx, coord_dim=dim, descending=False))
            return results
        elif reorder_type == "six":
            results = []
            for dim, desc_flag in zip([0, 1, 2]*2, [False, False, False, True, True, True]):
                results.append(composite_sort(coords, nodes, component_labels, orig_idx, coord_dim=dim, descending=desc_flag))
            return results

    # ========== split_parts > 1 ==========
    num_nodes_per_part = num_nodes // split_parts
    total_nodes_to_keep = num_nodes_per_part * split_parts
    perm_idx = torch.randperm(num_nodes, device=coords.device)[:total_nodes_to_keep]
    
    orig_idx = orig_idx.index_select(0, perm_idx)
    
    if split_parts == 2:
        groups = torch.split(perm_idx, num_nodes_per_part, dim=0)
        idx_groups = torch.split(orig_idx, num_nodes_per_part, dim=0)
        if reorder_type == "two":
            result1 = composite_sort(
                coords.index_select(0, groups[0]),
                nodes.index_select(0, groups[0]),
                component_labels.index_select(0, groups[0]),
                idx_groups[0],
                coord_dim=2, descending=False)
            result2 = composite_sort(
                coords.index_select(0, groups[1]),
                nodes.index_select(0, groups[1]),
                component_labels.index_select(0, groups[1]),
                idx_groups[1],
                coord_dim=2, descending=True)
            return [result1, result2]
        elif reorder_type == "six":
            results = []
            for dim in [0, 1, 2]:
                results.append(composite_sort(
                    coords.index_select(0, groups[0]),
                    nodes.index_select(0, groups[0]),
                    component_labels.index_select(0, groups[0]),
                    idx_groups[0],
                    coord_dim=dim, descending=False))
            for dim in [0, 1, 2]:
                results.append(composite_sort(
                    coords.index_select(0, groups[1]),
                    nodes.index_select(0, groups[1]),
                    component_labels.index_select(0, groups[1]),
                    idx_groups[1],
                    coord_dim=dim, descending=True))
            return results
    
    if split_parts == 3:
        groups = torch.split(perm_idx, num_nodes_per_part, dim=0)
        idx_groups = torch.split(orig_idx, num_nodes_per_part, dim=0)
        if reorder_type == "three":
            results = []
            for i, grp in enumerate(groups):
                results.append(composite_sort(
                    coords.index_select(0, grp),
                    nodes.index_select(0, grp),
                    component_labels.index_select(0, grp),
                    idx_groups[i],
                    coord_dim=i, descending=False))
            return results
        elif reorder_type == "six":
            asc_results = []
            desc_results = []
            for i, grp in enumerate(groups):
                asc_results.append(composite_sort(
                    coords.index_select(0, grp),
                    nodes.index_select(0, grp),
                    component_labels.index_select(0, grp),
                    idx_groups[i],
                    coord_dim=i, descending=False))
                desc_results.append(composite_sort(
                    coords.index_select(0, grp),
                    nodes.index_select(0, grp),
                    component_labels.index_select(0, grp),
                    idx_groups[i],
                    coord_dim=i, descending=True))
            return asc_results + desc_results
    
    if split_parts == 6:
        groups = torch.split(perm_idx, num_nodes_per_part, dim=0)
        idx_groups = torch.split(orig_idx, num_nodes_per_part, dim=0)
        results = []
        for i, grp in enumerate(groups):
            if i < 3:
                results.append(composite_sort(
                    coords.index_select(0, grp),
                    nodes.index_select(0, grp),
                    component_labels.index_select(0, grp),
                    idx_groups[i],
                    coord_dim=i, descending=False))
            else:
                results.append(composite_sort(
                    coords.index_select(0, grp),
                    nodes.index_select(0, grp),
                    component_labels.index_select(0, grp),
                    idx_groups[i],
                    coord_dim=i-3, descending=True))
        return results

