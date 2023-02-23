# cvt/util.py

"""A suite of common utility functions.

This module includes general utility functions used
by the sub-packages of the CVT library.

This module contains the following functions:

- `round_nearest(num, decimal=0)` - Rounds a floating point number to the nearest decimal place.
"""
import numpy as np
import torch

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



#   def center_image(img):
#       img = img.astype(np.float32)
#       var = np.var(img, axis=(0,1), keepdims=True)
#       mean = np.mean(img, axis=(0,1), keepdims=True)
#       return (img - mean) / (np.sqrt(var) + 0.00000001)
#   
#   
#   def scale_image(image, scale=1, interpolation='linear'):
#       if interpolation == 'linear':
#           return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#       if interpolation == 'nearest':
#           return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
#   
#   def scale_mvs_input(depths, confs, cams, scale=1):
#       views, height, width = depths.shape
#   
#       scaled_depths = []
#       scaled_confs = []
#   
#       for view in range(views):
#           scaled_depths.append(scale_image(depths[view], scale=scale, interpolation='linear'))
#           scaled_confs.append(scale_image(confs[view], scale=scale, interpolation='linear'))
#           cams[view] = scale_camera(cams[view], scale=scale)
#   
#       return np.asarray(scaled_depths), np.asarray(scaled_confs), cams
#   
#   def scale_gt(gt_depth, scale=1):
#       return scale_image(gt_depth, scale=scale, interpolation='linear')
#   
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
