# cvt/filtering.py

"""A suite of common filtering utilities.

This module includes several functions for filtering depth maps.

This module contains the following functions:

- `conf_filter(depth_map, conf_map, device, min_conf)` - Filters a map by confidence values above a minimum threshold.
- `topk_filter(depth_map, conf_map, device, percent)` - Filters a map by the top percentage of confidence values.
- `topk_strict_filter(depth_map, filter_prob, device, percent)` - Filters a map by the top percentage of confidence values.
"""

import torch

def conf_filter(depth_map: torch.Tensor, conf_map: torch.Tensor, device: str = 'cuda:0', min_conf: float = 0.8) -> Tuple(torch.Tensor, torch.Tensor):
    """Filters a map by confidence values above a minimum threshold.

    Parameters:
        depth_map:
        conf_map:
        device:
        min_conf:

    Returns:
        filtered_map:
        mask:
    """
    mask = (torch.ge(conf_map, min_conf)).to(torch.float32).to(device)
    return depth_map*mask, mask

def topk_filter(depth_map, conf_map, device='cuda:0', percent=0.3):
    """Filters a map by the top percentage of confidence values.

    Parameters:
        depth_map:
        conf_map:
        device:
        percentage:

    Returns:
        filtered_map:
        mask:
    """
    height, width = depth_map.shape

    # calculate k number of points to keep
    valid_map = torch.ne(conf_map, 0.0).to(torch.float32)
    valid_count = torch.sum(valid_map)
    k = int(percent * valid_count)

    # flatten and grab top-k indices
    filter_prob = conf_map.reshape(-1)
    (vals, indices) = torch.topk(filter_prob, k=k, dim=0)

    # get min confidence value
    min_conf = torch.min(vals)

    # filter by min conf value
    filt = (torch.ge(conf_map, min_conf)).to(torch.float32).to(device)

    return depth_map*filt, filt

def topk_strict_filter(depth_map, filter_prob, device='cuda:0', percent=0.3):
    """Filters a map by the top percentage of confidence values.

    Parameters:
        depth_map:
        conf_map:
        device:
        percentage:

    Returns:
        filtered_map:
        mask:
    """
    height, width = depth_map.shape

    # calculate k number of points to keep
    valid_map = torch.ne(filter_prob, 0.0).to(torch.float32)
    valid_count = torch.sum(valid_map)
    k = int(percent * valid_count)

    # flatten and grab top-k indices
    filter_prob = filter_prob.reshape(-1)
    (vals, indices) = torch.topk(filter_prob, k=k, dim=0)

    # calculate the row and column given each index
    row_indices = torch.div(indices, width, rounding_mode="floor").unsqueeze(-1)
    col_indices = torch.remainder(indices, width).unsqueeze(-1)

    # concatenate the [r,c] indices into a single tensor
    indices = torch.cat((row_indices, col_indices), dim=1)
    filt = torch.zeros((height,width), dtype=torch.uint8).to(device)

    # set top-k indices to 1
    for r,c in indices:
        filt[r,c] = 1

    return depth_map*filt, filt
