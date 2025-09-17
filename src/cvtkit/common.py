# cvt/util.py

"""A suite of common utility functions.

This module includes general utility functions used
by the sub-packages of the CVTkit library.
"""

import numpy as np
import cv2
import random
from math import exp

import torch
import torch.nn.functional as F

from typing import Tuple, Any
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module


def build_coords_list(H: int, W: int, batch_size: int, device: str) -> torch.Tensor:
    """Constructs an batched index list of pixel coordinates.

    Parameters:
        H: Height of the pixel grid.
        W: Width of the pixel grid.
        batch_size: Number of batches.
        device: GPU device identifier.

    Returns:
        The index list of shape [batch_size, H*W, 2]
    """
    indices_h = torch.linspace(0, H, H, dtype=torch.int64)
    indices_w = torch.linspace(0, W, W, dtype=torch.int64)
    indices_h, indices_w = torch.meshgrid(indices_h, indices_w)
    indices = torch.stack([indices_h, indices_w], dim=-1).to(torch.int64).to(device)
    indices = indices.reshape(1, -1, 2).repeat(batch_size, 1, 1)
    return indices


def build_depth_pyramid(
    depth: NDArray[np.float32], levels: int
) -> dict[int, NDArray[Any]]:
    """ """
    h, w = depth.shape

    depths: dict[int, NDArray[Any]] = {(levels - 1): depth.reshape(1, h, w)}
    for i in range(1, levels):
        size = (int(w // (2**i)), int(h // (2**i)))
        d = np.asarray(
            cv2.resize(src=depth, dsize=size, interpolation=cv2.INTER_LINEAR)
        )

        depths[levels - 1 - i] = d.reshape(1, size[1], size[0])

    return depths


def build_labels(depth: Tensor, hypotheses: Tensor) -> tuple[Tensor, Tensor]:
    """ """
    bin_radius = (hypotheses[:, 1] - hypotheses[:, 0]) / 2.0
    target_bin_dist = torch.abs(depth - hypotheses)
    target_labels = torch.where(target_bin_dist <= bin_radius.unsqueeze(1), 1.0, 0.0)
    mask = torch.where(target_labels.sum(dim=1) > 0, 1.0, 0.0) * torch.where(
        depth.squeeze(1) > 0, 1.0, 0.0
    )
    target_labels = torch.argmax(target_labels, dim=1)

    return target_labels.to(torch.int64), mask.to(torch.float32)


def crop_image(
    image: NDArray[Any] | Tensor, crop_row: int, crop_col: int, scale: float
) -> NDArray[Any] | Tensor:
    """ """
    _, height, width = image.shape

    start_row = crop_row
    end_row = int(crop_row + (scale * height))
    start_col = crop_col
    end_col = int(crop_col + (scale * width))

    return image[:, start_row:end_row, start_col:end_col]


def freeze_model_weights(model: Module) -> None:
    """ """
    model.requires_grad_(False)


def gaussian(window_size: int, sigma: float):
    """ """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def laplacian_pyramid_th(image: Tensor, tau: float) -> Tensor:
    """Computes the Laplacian pyramid of an image.

    Parameters:
        image: 2D map to compute Laplacian over.
        tau: Laplacian region threshold.

    Returns:
        The map of the Laplacian regions.
    """
    batch_size, _, h, w = image.shape
    levels = 4

    # build gaussian pyramid
    pyr = [image]
    for l in range(levels):
        pyr.append(F.interpolate(input=pyr[-1], scale_factor=0.5, mode="bilinear"))

    # compute laplacian pyramid (differance between gaussian pyramid levels)
    all_diff_mask = torch.zeros_like(image)
    for l in range(levels, 0, -1):
        region_id = levels - l + 1

        diff = (
            torch.abs(
                F.interpolate(pyr[l], scale_factor=2, mode="bilinear") - pyr[l - 1]
            )
        ).mean(
            dim=1, keepdim=True
        )  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        diff = F.interpolate(
            input=diff, size=(h, w), mode="bilinear"
        )  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        diff_mask = torch.where(
            diff > tau, 1, 0
        )  # pyright: ignore[reportUnknownArgumentType]

        diff_mask = diff_mask.reshape(batch_size, h, w, 1)

        if l == levels:
            all_diff_mask = diff_mask * region_id
        else:
            all_diff_mask = torch.where(diff_mask == 1, region_id, all_diff_mask)

    return all_diff_mask.reshape(batch_size, 1, h, w)


def laplacian_pyramid(image: torch.Tensor) -> torch.Tensor:
    """Computes the Laplacian pyramid of an image.

    Parameters:
        image: 2D map to compute Laplacian over.
        tau: Laplacian region threshold.

    Returns:
        The map of the Laplacian regions.
    """
    batch_size, c, h, w = image.shape
    levels = 4

    # build gaussian pyramid
    pyr = [image]
    for l in range(levels):
        pyr.append(F.interpolate(pyr[-1], scale_factor=0.5, mode="bilinear"))

    # compute laplacian pyramid (differance between gaussian pyramid levels)
    all_diff = torch.zeros_like(image)
    for l in range(levels, 0, -1):
        region_id = levels - l + 1

        diff = (
            torch.abs(
                F.interpolate(pyr[l], scale_factor=2, mode="bilinear") - pyr[l - 1]
            )
        ).mean(dim=1, keepdim=True)
        diff = F.interpolate(diff, size=(h, w), mode="bilinear")
        diff = diff.reshape(batch_size, h, w, 1)

        if l == levels:
            all_diff = diff
        else:
            all_diff += diff

    return all_diff.reshape(batch_size, 1, h, w)


def cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Computes the cosine similarity between two tensors.

    Parameters:
        t1: First tensor.
        t2: Second tensor.

    Returns:
       The cosine similarity between the two tensors.
    """
    assert t1.shape == t2.shape
    similarity = torch.abs(F.cosine_similarity(t1, t2, dim=1).unsqueeze(1))

    if len(t1.shape) == 5:
        B, _, D, H, W = t1.shape
        assert similarity.shape == (B, 1, D, H, W)
    elif len(t1.shape) == 4:
        B, _, H, W = t1.shape
        assert similarity.shape == (B, 1, H, W)
    else:
        raise Exception(
            "Can only compute cosine similarity with 4 or 5 dimension tensors"
        )

    return similarity


def groupwise_correlation(
    t1: torch.Tensor, t2: torch.Tensor, num_groups: int
) -> torch.Tensor:
    """Computes the Group-Wise Correlation (GWC) between two tensors.

    Parameters:
        t1: First tensor.
        t2: Second tensor.
        num_groups: Number of groups.

    Returns:
       The Group-Wise Correlation (GWC) between the two tensors.
    """
    assert t1.shape == t2.shape
    if len(t1.shape) == 5:
        B, C, D, H, W = t1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        correlation = (
            (t1 * t2).view([B, num_groups, channels_per_group, D, H, W]).mean(dim=2)
        )
        assert correlation.shape == (B, num_groups, D, H, W)
    elif len(t1.shape) == 4:
        B, C, H, W = t1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        correlation = (
            (t1 * t2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
        )
        assert correlation.shape == (B, num_groups, H, W)
    else:
        raise Exception("Can only compute GWC with 4 or 5 dimension tensors")

    return correlation


def inverse_sigmoid(x, scale=1):
    return torch.log(x / (1 - x)) / scale


def non_zero_std(
    maps: torch.Tensor, device: str, dim: int = 1, keepdim: bool = False
) -> torch.Tensor:
    """Computes the standard deviation of all non-zero values in an input Tensor along the given dimension.

    Parameters:
        maps:
        device:
        keepdim:

    Returns:
        The standard deviation of the non-zero elements of the input map.
    """
    batch_size, views, height, width = maps.shape
    valid_map = torch.ne(maps, 0.0).to(torch.float32).to(device)
    valid_count = torch.sum(valid_map, dim=1, keepdim=keepdim) + 1e-7
    mean = (
        torch.div(torch.sum(maps, dim=1, keepdim=keepdim), valid_count)
        .reshape(batch_size, 1, height, width)
        .repeat(1, views, 1, 1)
    )
    mean = torch.mul(valid_map, mean)

    std = torch.sub(maps, mean)
    std = torch.square(std)
    std = torch.sum(std, dim=1, keepdim=keepdim)
    std = torch.div(std, valid_count)
    std = torch.sqrt(std)

    return std


def normalize(
    data: NDArray[Any] | Tensor,
    mean: float | None = None,
    std: float | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
):
    """ """
    if min_val is None:
        min_val = float(data.min())
    if max_val is None:
        max_val = float(data.max())

    data = (data - min_val) / (max_val - min_val + 1e-10)

    if mean != None and std != None:
        data = (data - mean) / std

    return data


def parameters_count(net, name, do_print=True):
    """

    Parameters:

    Returns:
    """
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    if do_print:
        print(f"#params {name}: {(params/1e6):0.3f} M")
    return params


def print_gpu_mem() -> None:
    """Prints the current unallocated memory of the GPU."""
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = t - (a + r)
    print("Free: {:0.4f} GB".format(f / (1024 * 1024 * 1024)))


def round_nearest(num: float, decimal: int = 0) -> int:
    """Rounds a floating point number to the nearest decimal place.

    Args:
        num: Float to be rounded.
        decimal: Decimal place to round to.

    Returns:
        The given number rounded to the nearest decimal place.

    Examples:
        >>> round_nearest(11.1)
        11
        >>> round_nearest(15.7)
        16
        >>> round_nearest(2.5)
        2
        >>> round_nearest(3.5)
        3
        >>> round_nearest(14.156, 1)
        14.2
        >>> round_nearest(15.156, 1)
        15.2
        >>> round_nearest(15.156, 2)
        15.16
    """

    return np.round(num + 10 ** (-len(str(num)) - 1), decimal)


def scale_camera(cam: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Scales a camera intrinsic parameters.

    Parameters:
        cam: Input camera to be scaled.
        scale: Scale factor.

    Returns:
        The scaled camera.
    """
    new_cam = np.copy(cam)
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam


def scale_image(
    image: np.ndarray, scale: float = 1.0, interpolation: str = "linear"
) -> NDArray[np.float32]:
    """Scales an input pixel grid.

    Parameters:
        image: Input image to be scaled.
        scale: Scale factor.
        interpolation: Interpolation technique to be used.

    Returns:
        The scaled image.
    """
    if interpolation == "linear":
        return cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
    elif interpolation == "nearest":
        return cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)
    else:
        raise Exception(f"ERROR: unknown interpolation mode '{interpolation}'")


def scale_mvs_data(
    depths: np.ndarray,
    confs: np.ndarray,
    cams: np.ndarray,
    scale: float = 1.0,
    interpolation: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scales input depth maps, confidence maps, and cameras.

    Parameters:
        depths: Input depth maps to be scaled.
        confs: Input confidence maps to be scaled.
        cams: Input cameras to be scaled
        scale: Scale factor.
        interpolation: Interpolation technique.

    Returns:
        scaled_depths: The scaled depth maps.
        scaled_confs: The scaled confidence maps.
        cams: The scaled cameras.
    """
    views, height, width = depths.shape

    scaled_depths = []
    scaled_confs = []

    for view in range(views):
        scaled_depths.append(
            scale_image(depths[view], scale=scale, interpolation=interpolation)
        )
        scaled_confs.append(
            scale_image(confs[view], scale=scale, interpolation=interpolation)
        )
        cams[view] = scale_camera(cams[view], scale=scale)

    return np.asarray(scaled_depths), np.asarray(scaled_confs), cams


def set_random_seed(seed):
    """

    Parameters:

    Returns:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_gpu(data: dict, device: str) -> None:
    """Loads a dictionary of elements onto the GPU device.

    Parameters:
        data: Dictionary to be loaded.
        device: GPU device identifier.
    """

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            data[key] = val.cuda(device, non_blocking=True)
        if isinstance(val, dict):
            for k1, v1 in val.items():
                if isinstance(v1, torch.Tensor):
                    data[key][k1] = v1.cuda(device, non_blocking=True)
    return


def top_k_hypothesis_selection(grid, k, prev_hypos, prev_hypo_coords, prev_intervals):
    selected_prob, selected_idx = torch.topk(grid, k=k, dim=2)
    selected_hypos = torch.gather(prev_hypos, dim=2, index=selected_idx)
    selected_intervals = torch.gather(prev_intervals, dim=2, index=selected_idx)
    selected_intervals = selected_intervals / 2
    selected_coords = torch.gather(prev_hypo_coords, dim=2, index=selected_idx)
    # subdivide hypos
    upper_new_hypos = selected_hypos + selected_intervals / 2
    lower_new_hypos = selected_hypos - selected_intervals / 2
    new_hypos = torch.cat((upper_new_hypos, lower_new_hypos), dim=2)
    new_hypos = torch.repeat_interleave(new_hypos, 2, dim=3)
    new_hypos = torch.repeat_interleave(new_hypos, 2, dim=4)

    # subdivide coords
    upper_new_coords = selected_coords * 2 + 1
    lower_new_coords = selected_coords * 2
    new_coords = torch.cat((upper_new_coords, lower_new_coords), dim=2)
    new_coords = torch.repeat_interleave(new_coords, 2, dim=3)
    new_coords = torch.repeat_interleave(new_coords, 2, dim=4)

    # subdivide intervals
    selected_intervals = torch.cat(
        (selected_intervals, selected_intervals), dim=2
    )  # Dx2
    selected_intervals = torch.repeat_interleave(selected_intervals, 2, dim=3)  # Hx2
    new_intervals = torch.repeat_interleave(selected_intervals, 2, dim=4)  # Wx2

    return new_hypos, new_coords, new_intervals
