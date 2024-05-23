# cvt/util.py

"""A suite of common utility functions.

This module includes general utility functions used
by the sub-packages of the CVT library.

This module contains the following functions:

- `non_zero_std(maps, device, dim, keepdim)` - Computes the standard deviation of all non-zero values in an input Tensor along the given dimension.
- `print_gpu_mem()` - Prints the current unallocated memory of the GPU.
- `round_nearest(num, decimal=0)` - Rounds a floating point number to the nearest decimal place.
- `scale_image(image, scale, interpolation)` - Scales an input pixel grid.
"""

import torch
import numpy as np
import cv2
import random

from typing import Tuple

from camera import Z_from_disp, intrinsic_pyramid

def print_csv(data):
    for i,d in enumerate(data):
        if i==len(data)-1:
            print(f"{d:6.4f}")
        else:
            print(f"{d:6.4f}", end=",")

def to_gpu(data, device):
    no_gpu_list = ["index", "filenames", "num_frame"]

    for key,val in data.items():
        if (key not in no_gpu_list):
            if key == "gbinet_cams" or key == "binary_tree":
                for k,v in data[key].items():
                        data[key][k] = v.cuda(device, non_blocking=True)
            else:
                data[key] = val.cuda(device, non_blocking=True)

def build_coords_list(H, W, batch_size, device):
    indices_h = torch.linspace(0, H, H, dtype=torch.int64)
    indices_w = torch.linspace(0, W, W, dtype=torch.int64)
    indices_h, indices_w = torch.meshgrid(indices_h, indices_w)
    indices = torch.stack([indices_h, indices_w], dim=-1).to(torch.int64).to(device)
    indices = indices.reshape(1,-1,2).repeat(batch_size,1,1)
    return indices

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parameters_count(net, name, do_print=True):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    if do_print:
        print(f"#params {name}: {(params/1e6):0.3f} M")
    return params



def top_k_hypothesis_selection(grid, k, prev_hypos, prev_hypo_coords, prev_intervals):
    selected_prob, selected_idx = torch.topk(grid,k=k,dim=2)
    selected_hypos = torch.gather(prev_hypos, dim=2, index=selected_idx)
    selected_intervals = torch.gather(prev_intervals, dim=2, index=selected_idx)
    selected_intervals = selected_intervals/2
    selected_coords = torch.gather(prev_hypo_coords, dim=2, index=selected_idx)
    # subdivide hypos
    upper_new_hypos = selected_hypos+selected_intervals/2
    lower_new_hypos = selected_hypos-selected_intervals/2
    new_hypos = torch.cat((upper_new_hypos,lower_new_hypos),dim=2)
    new_hypos = torch.repeat_interleave(new_hypos,2,dim=3)
    new_hypos = torch.repeat_interleave(new_hypos,2,dim=4)

    # subdivide coords
    upper_new_coords = selected_coords*2+1
    lower_new_coords = selected_coords*2
    new_coords = torch.cat((upper_new_coords,lower_new_coords),dim=2)
    new_coords = torch.repeat_interleave(new_coords,2,dim=3)
    new_coords = torch.repeat_interleave(new_coords,2,dim=4)

    # subdivide intervals
    selected_intervals = torch.cat((selected_intervals,selected_intervals),dim=2) # Dx2
    selected_intervals = torch.repeat_interleave(selected_intervals,2,dim=3) # Hx2
    new_intervals = torch.repeat_interleave(selected_intervals,2,dim=4) # Wx2

    return new_hypos, new_coords, new_intervals

def groupwise_correlation(v1, v2, num_groups):
    assert(v1.shape==v2.shape)
    if (len(v1.shape) == 5):
        B, C, D, H, W = v1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        cost_volume = (v1 * v2).view([B, num_groups, channels_per_group,D, H, W]).mean(dim=2)
        assert cost_volume.shape == (B, num_groups, D, H, W)
    elif (len(v1.shape) == 4):
        B, C, H, W = v1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        cost_volume = (v1 * v2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
        assert cost_volume.shape == (B, num_groups, H, W)
    else:
        print("Can only compute GWC with 4 or 5 dimension tensors")
        sys.exit()
    return cost_volume

def cosine_similarity(v1, v2, num_groups):
    assert(v1.shape==v2.shape)
    B, C, D, H, W = v1.shape
    cost_volume = torch.abs(F.cosine_similarity(v1,v2,dim=1).unsqueeze(1))
    assert(cost_volume.shape == (B, 1, D, H, W))

    return cost_volume

def non_zero_std(maps: torch.Tensor, device: str, dim: int = 1, keepdim: bool = False) -> torch.Tensor:
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
    valid_count = torch.sum(valid_map, dim=1, keepdim=keepdim)+1e-7
    mean = torch.div(torch.sum(maps,dim=1, keepdim=keepdim), valid_count).reshape(batch_size, 1, height, width).repeat(1,views,1,1)
    mean = torch.mul(valid_map, mean)

    std = torch.sub(maps, mean)
    std = torch.square(std)
    std = torch.sum(std, dim=1, keepdim=keepdim)
    std = torch.div(std, valid_count)
    std = torch.sqrt(std)

    return std

def print_gpu_mem() -> None:
    """Prints the current unallocated memory of the GPU.
    """
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = t- (a+r)
    print("Free: {:0.4f} GB".format(f/(1024*1024*1024)))

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

    return round(num+10**(-len(str(num))-1), decimal)


#   def center_image(img):
#       img = img.astype(np.float32)
#       var = np.var(img, axis=(0,1), keepdims=True)
#       mean = np.mean(img, axis=(0,1), keepdims=True)
#       return (img - mean) / (np.sqrt(var) + 0.00000001)
#   
#   

def compute_laplacian_pyr(image, levels=4):
    image = torch.movedim(image, (0,1,2,3), (0,2,3,1))
    batch_size, c, h, w = image.shape

    # for visualiation
    color = torch.zeros(levels+1,1,1,3).to(image)
    color[0,0,0,0], color[0,0,0,1], color[0,0,0,2] = 153, 0, 0 #red
    color[1,0,0,0], color[1,0,0,1], color[1,0,0,2] = 255, 128, 0 #orange
    color[2,0,0,0], color[2,0,0,1], color[2,0,0,2] = 0, 255, 0 #green
    color[3,0,0,0], color[3,0,0,1], color[3,0,0,2] = 0, 255, 255 #cyan
    color[4,0,0,0], color[4,0,0,1], color[4,0,0,2] = 51, 0, 102 #dark purple

    # to ignore changes near image border
    crop_mask = torch.zeros(batch_size, h, w).to(image)
    crop_mask[0, 20:h-20, 20:w-20] = 1.0

    # build gaussian pyramid
    pyr = [image]
    for l in range(levels):
        pyr.append(F.interpolate(pyr[-1], scale_factor=0.5, mode="bilinear"))

    # compute laplacian pyramid (differance between gaussian pyramid levels)
    laplacian = torch.zeros(levels, batch_size, 1, h, w).to(image)
    for l in range(levels, 0, -1):
        diff = (torch.abs(F.interpolate(pyr[l], scale_factor=2, mode="bilinear") - pyr[l-1])).mean(dim=1, keepdim=True)
        diff = F.interpolate(diff, size=(h, w), mode="bilinear")
        diff *= crop_mask

        d_th = diff.mean()
        #dmin = diff.min()
        #dmax = diff.max()
        #diff = (diff-dmin)/(dmax-dmin)
        diff = torch.where(diff > d_th, 1, 0)
        laplacian[l-1] = diff

        diff = diff.reshape(h,w,1)
        color_diff = diff.repeat(1,1,3) * color[l-1]
        cv2.imwrite(f"./log/laplace_{l-1}.png", color_diff.flip(dims=[-1]).detach().cpu().numpy())

        if l==levels:
            all_diff = color_diff
        else:
            all_diff = torch.where(diff==1, color[l-1,0,0], all_diff)

    #cv2.imwrite("./log/laplace.png", (laplacian.sum(dim=0))[0,0].detach().cpu().numpy()*50)
    all_diff = torch.where(all_diff.sum(dim=-1,keepdim=True)==0, color[-1,0,0], all_diff)
    cv2.imwrite("./log/laplace.png", all_diff.flip(dims=[-1]).detach().cpu().numpy())
    cv2.imwrite("./log/image.png", torch.movedim(image[0].flip(dims=[0]), (0,1,2), (2,0,1)).detach().cpu().numpy()*255)
    sys.exit()


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

def scale_image(image: np.ndarray, scale: float = 1.0, interpolation: str = "linear") -> np.ndarray:
    """Scales an input pixel grid.

    Parameters:
        image: Input image to be scaled.
        scale: Scale factor.
        interpolation: Interpolation technique to be used.

    Returns:
        The scaled image.
    """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def scale_mvs_data(depths: np.ndarray, confs: np.ndarray, cams: np.ndarray, scale: float = 1.0, interpolation: str = "linear") -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
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
        scaled_depths.append(scale_image(depths[view], scale=scale, interpolation=interpolation))
        scaled_confs.append(scale_image(confs[view], scale=scale, interpolation=interpolation))
        cams[view] = scale_camera(cams[view], scale=scale)

    return np.asarray(scaled_depths), np.asarray(scaled_confs), cams


#   def crop_mvs_input(images, cams, depth_image=None, max_w=0, max_h=0):
#       # crop images and cameras
#       for view in range(FLAGS.view_num):
#           h, w = images[view].shape[0:2]
#           new_h = h
#           new_w = w
#           if new_h > FLAGS.max_h:
#               new_h = FLAGS.max_h
#           else:
#               new_h = int(math.ceil(h / FLAGS.base_image_size) * FLAGS.base_image_size)
#           if new_w > FLAGS.max_w:
#               new_w = FLAGS.max_w
#           else:
#               new_w = int(math.ceil(w / FLAGS.base_image_size) * FLAGS.base_image_size)
#   
#           if max_w > 0:
#               new_w = max_w
#           if max_h > 0:
#               new_h = max_h
#   
#           start_h = int(math.ceil((h - new_h) / 2))
#           start_w = int(math.ceil((w - new_w) / 2))
#           finish_h = start_h + new_h
#           finish_w = start_w + new_w
#           images[view] = images[view][start_h:finish_h, start_w:finish_w]
#           cams[view][1][0][2] = cams[view][1][0][2] - start_w
#           cams[view][1][1][2] = cams[view][1][1][2] - start_h
#   
#           # crop depth image
#           if not depth_image is None and view == 0:
#               depth_image = depth_image[start_h:finish_h, start_w:finish_w]
#   
#       if not depth_image is None:
#           return images, cams, depth_image
#       else:
#           return images, cams
#   
#   def mask_depth_image(depth_image, min_depth, max_depth):
#       ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
#       ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
#       depth_image = np.expand_dims(depth_image, 2)
#       return depth_image
