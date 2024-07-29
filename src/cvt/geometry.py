# cvt/geometry.py

"""Module including geometric routines.

This module contains the following functions:

- `downsample_cloud(cloud, min_point_dist)` - Downsamples a point cloud enforcing a minumum point spacing.
- `essential_from_features(src_image_file, tgt_image_file, K)` - Computes the essential matrix between two images using image features.
- `fundamental_from_KP(K, P_src, P_tgt)` - Computes the fundamental matrix between two images using camera parameters.
- `fundamental_from_features(src_image_file, tgt_image_file)` - Computes the fundamental matrix between two images using image features.
- `geometric_consistency_error(src_depth, src_cam, tgt_depth, tgt_cam)` - .
- `geometric_consistency_mask(src_depth, src_cam, tgt_depth, tgt_cam, pixel_th)` - Computes the geometric consistency mask between a source and target depth map.
- `homography(src_image_file, tgt_image_file)` - Computes a homography transformation between two images using image features.
- `homography_warp(cfg,features, level, ref_in, src_in, ref_ex, src_ex, depth_hypos, gwc_groups, va_net, vis_weights, aggregation)` - Performs homography warping to create a Plane Sweeping Volume (PSV).
- `match_features(src_image, tgt_image, max_features)` - Computer matching ORB features between a pair of images.
- `plane_coords(K, P, depth_hypos, H, W)` - .
- `points_from_depth(depth, cam)` - Creates a point array from a single depth map.
- `project_depth_map(depth, cam, mask)` - Projects a depth map into a list of 3D points.
- `project_renderer(renderer, K, P, width, height)` - Projects the scene in an Open3D Offscreen Renderer to the 2D image plane.
- `render_custom_values(points, values, image_shape, cam)` - Renders a point cloud into a 2D camera plane using custom values for each pixel.
- `render_point_cloud(cloud, cam, width, height)` - Renders a point cloud into a 2D image plane.
- `reproject(src_depth, src_cam, tgt_depth, tgt_cam)` - Computes the re-projection depth values and pixel indices between two depth maps.
- `sample_volume(volume, z_vals, coords, H, W, near_depth, far_depth, inv_depth)` - .
- `soft_hypothesis(data, target_hypo, focal_length, min_hypo, max_hypo, M, delta_in)` - .
- `visibility(depths, K, Ps, vis_th, levels)` - .
- `visibility_mask(src_depth, src_cam, depth_files, cam_files, src_ind, pixel_th)` - Computes a visibility mask between a provided source depth map and list of target depth maps.
- `uniform_hypothesis(cfg, ref_in, src_in, ref_ex, src_ex, depth_min, depth_max, img_height, img_width, nhypothesis_init, inv_depth)` - .
"""

import numpy as np
import sys
import cv2
import open3d as o3d
from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
import torchvision.transforms as tvt
from torch.cuda.amp import autocast

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from camera import intrinsic_pyramid, Z_from_disp
from common import groupwise_correlation
from io import *


def downsample_cloud(cloud, min_point_dist):
    """Downsamples a point cloud enforcing a minumum point spacing.
    Parameters:
        cloud: Point cloud to be decimated.
        min_point_dist: minimum point spacing to enforce.

    Returns:
        The downsampled point cloud.
    """
    return cloud.voxel_down_sample(voxel_size=min_point_dist)

def get_epipolar_inds(x0, y0, x1, y1, x_lim, y_lim, max_patches):
    dx = x1-x0
    dy = y1-y0
    yi = torch.ones_like(y0)

    negative_mask = torch.where(dy < 0, -1, 1)
    yi *= negative_mask
    dy *= negative_mask

    D = (2*dy) - dx
    y = y0
    x = x0

    batch,h,w = x.shape

    epipolar_grid = torch.zeros((batch, max_patches, h, w, 2)).to(x0)
    for i in range(max_patches):
        # build valid indices mask
        valid_mask = torch.where((x < x_lim).to(torch.bool) & (x >= 0).to(torch.bool), 1, 0)
        valid_mask *= torch.where((y < y_lim).to(torch.bool) & (y >= 0).to(torch.bool), 1, 0)
        valid_mask = valid_mask.unsqueeze(-1).repeat(1,1,1,2)
        
        # stack xy and apply valid indices mask
        xy = torch.stack([x,y],dim=-1)
        epipolar_grid[:,i,:,:,:] = (xy*valid_mask) - (1-valid_mask)

        mask = torch.where(D > 0, 1, 0)
        y = (y+yi)*mask + y*(1-mask)
        D = ((D + (2*(dy-dx)))*mask) + ((D + (2*dy))*(1-mask))
        x += 1

    return epipolar_grid[:,:,:,:,0], epipolar_grid[:,:,:,:,1]

def get_epipolar_inds_low(x0, y0, x1, y1):
    dx = x1-x0
    dy = y1-y0
    yi=1
    if dy < 0:
        yi=-1
        dy=-dy

    D = (2*dy) - dx
    y=y0

    xy = []
    for x in range(x0,x1+1):
        xy.append([x,y])

        if D > 0:
            y = y + yi
            D = D + (2*(dy-dx))
        else:
            D = D + (2*dy)

    xy = np.asarray(xy).astype(np.int32)
    return xy[:,0], xy[:,1]

def get_epipolar_inds_high(x0, y0, x1, y1):
    dx = x1-x0
    dy = y1-y0
    xi=1
    if dx < 0:
        xi=-1
        dx=-dx

    D = (2*dx) - dy
    x=x0

    xy = []
    for y in range(y0,y1+1):
        xy.append([x,y])

        if D > 0:
            x = x + xi
            D = D + (2*(dx-dy))
        else:
            D = D + (2*dx)

    xy = np.asarray(xy).astype(np.int32)
    return xy[:,0], xy[:,1]

def epipolar_patch_retrieval(imgs, intrinsics, extrinsics, patch_size):
    batch_size, _, _, height, width = imgs.shape
    K = intrinsics[:,0]
    P_src = extrinsics[:,0]
    half_patch_size = patch_size//2

    x_flat = torch.arange((half_patch_size),width+1)[::patch_size].to(imgs)
    y_flat = torch.arange((half_patch_size),height+1)[::patch_size].to(imgs)

    xgrid, ygrid = torch.meshgrid([x_flat,y_flat], indexing="xy")
    xy = torch.stack([xgrid, ygrid, torch.ones_like(xgrid)], dim=-1) # [patched_height, patch_width, 3]
    patched_height, patched_width, _ = xy.shape
    xy = xy.unsqueeze(0).repeat(batch_size, 1, 1, 1).unsqueeze(-1) # [batch_size, patched_height, patch_width, 3, 1]

    max_patches = patched_height+patched_width-1

    for i in range(1,imgs.shape[1]):
        x_lim = (torch.ones((batch_size, patched_height, patched_width)) * patched_width).to(imgs)
        y_lim = (torch.ones((batch_size, patched_height, patched_width)) * patched_height).to(imgs)
    
        P_tgt = extrinsics[:,i]
        Fm = fundamental_from_KP(K, P_src, P_tgt)
        Fm = Fm.reshape(batch_size, 1, 1, 3, 3).repeat(1, xy.shape[1], xy.shape[2], 1, 1) # [batch_size, patched_height, patch_width, 3, 1]
        line = torch.matmul(Fm,xy).squeeze(-1) # [batch_size, patched_height, patch_width, 3]
        
        ## Start Point ##
        # initial x coordinate, comput y corrdinate
        x0 = x_flat[0].reshape(1,1,1).repeat(batch_size, patched_height, patched_width)
        y0 = (-(line[:,:,:,0]/line[:,:,:,1])*x0) - (line[:,:,:,2]/line[:,:,:,1])
        # check for invalid y coordinates
        y_mask_lt = torch.where(y0 < 0, 1, 0)
        y_mask_gt = torch.where(y0 >= height, 1, 0)
        y_mask_out = y_mask_lt+y_mask_gt
        # adjust for invalid y coordinates
        y0 = y0*(1-y_mask_lt) + y_flat[0]*y_mask_lt
        y0 = y0*(1-y_mask_gt) + y_flat[-1]*y_mask_gt
        x0 = (x0 * (1-y_mask_out)) + (((-(line[:,:,:,1]/line[:,:,:,0])*y0) - (line[:,:,:,2]/line[:,:,:,0])) * (y_mask_out))
        # if x coordinate is invalid
        valid_mask = torch.where((x0 >= 0).to(torch.bool) & (x0 < width).to(torch.bool), 1, 0)
        x0 = (x0*valid_mask) - (1-valid_mask)
        y0 = (y0*valid_mask) - (1-valid_mask)

        ## End Point ##
        # initial x coordinate, comput y corrdinate
        x1 = x_flat[-1].reshape(1,1,1).repeat(batch_size, patched_height, patched_width)
        y1 = (-(line[:,:,:,0]/line[:,:,:,1])*x1) - (line[:,:,:,2]/line[:,:,:,1])
        # check for invalid y coordinates
        y_mask_lt = torch.where(y1 < 0, 1, 0)
        y_mask_gt = torch.where(y1 >= height, 1, 0)
        y_mask_out = y_mask_lt+y_mask_gt
        # adjust for invalid y coordinates
        y1 = y1*(1-y_mask_lt) + y_flat[0]*y_mask_lt
        y1 = y1*(1-y_mask_gt) + y_flat[-1]*y_mask_gt
        x1 = (x1 * (1-y_mask_out)) + (((-(line[:,:,:,1]/line[:,:,:,0])*y1) - (line[:,:,:,2]/line[:,:,:,0])) * (y_mask_out))
        # if x coordinate is invalid
        valid_mask = torch.where((x1 >= 0).to(torch.bool) & (x1 < width).to(torch.bool), 1, 0)
        x1 = (x1*valid_mask) - (1-valid_mask)
        y1 = (y1*valid_mask) - (1-valid_mask)

        # convert image indices into patch indices
        x0 = (x0-(half_patch_size))//patch_size
        x1 = (x1-(half_patch_size))//patch_size
        y0 = (y0-(half_patch_size))//patch_size
        y1 = (y1-(half_patch_size))//patch_size

        # compute x and y slopes
        slope_x = torch.abs(x1-x0)
        slope_y = torch.abs(y1-y0)

        # flip x's and y's depending on slope
        small_slope_mask = torch.where(slope_y < slope_x, 1, 0)
        x0_temp = x0*small_slope_mask + y0*(1 - small_slope_mask)
        y0 = y0*small_slope_mask + x0*(1 - small_slope_mask)
        x0 = x0_temp
        x1_temp = x1*small_slope_mask + y1*(1 - small_slope_mask)
        y1 = y1*small_slope_mask + x1*(1 - small_slope_mask)
        x1 = x1_temp
        x_lim_temp = x_lim*small_slope_mask + y_lim*(1 - small_slope_mask)
        y_lim = y_lim*small_slope_mask + x_lim*(1 - small_slope_mask)
        x_lim = x_lim_temp
        
        # flip start and end points so start is smaller
        small_end_mask = torch.where(x1 < x0, 1, 0)
        x0_temp = x0*(1-small_end_mask) + x1*small_end_mask
        x1 = x1*(1-small_end_mask) + x0*small_end_mask
        x0 = x0_temp
        y0_temp = y0*(1-small_end_mask) + y1*small_end_mask
        y1 = y1*(1-small_end_mask) + y0*small_end_mask
        y0 = y0_temp

        # grab nearest patch indices
        x_grid, y_grid = get_epipolar_inds(x0, y0, x1, y1, x_lim, y_lim, max_patches)

        # flip x and y indices back where necessary (using small_slope_mask)
        small_slope_mask = small_slope_mask.reshape(batch_size,1,patched_height,patched_width).repeat(1,max_patches,1,1)
        x_grid_temp = x_grid*small_slope_mask + y_grid*(1 - small_slope_mask)
        y_grid = y_grid*small_slope_mask + x_grid*(1 - small_slope_mask)
        x_gird = x_grid_temp
        epipolar_grid = torch.stack([x_grid,y_grid], dim=-1)

        # convert patch indices into image indices
        epipolar_grid = epipolar_grid*patch_size + (half_patch_size)
        epipolar_grid = torch.where(epipolar_grid < 0, -1, epipolar_grid)

        # duplicate patch center indices over entire patch
        epipolar_grid = torch.repeat_interleave(epipolar_grid, patch_size, dim=2)
        epipolar_grid = torch.repeat_interleave(epipolar_grid, patch_size, dim=3)

        # apply center offset matrix
        valid_mask = torch.where(epipolar_grid >= 0, 1, 0)
        patch_offset = torch.arange(-half_patch_size,half_patch_size).to(imgs)
        x_offset, y_offset = torch.meshgrid([patch_offset,patch_offset], indexing="xy")
        patch_offset = torch.stack([x_offset,y_offset],dim=-1)
        patch_offset = torch.tile(patch_offset, (patched_height, patched_width,1))
        patch_offset = patch_offset.reshape(1,1,height,width,2).repeat(batch_size,max_patches,1,1,1)
        epipolar_grid += patch_offset

        x = int(epipolar_grid[0,20,200,200,0])
        y = int(epipolar_grid[0,20,200,200,1])
        print(x, y)
        print(imgs[:,i][0,:,y,x])
        img_patches = F.grid_sample(imgs[:,i],
                                    epipolar_grid.reshape(batch_size,max_patches,height*width,2),
                                    mode="nearest",
                                    padding_mode="zeros")
        img_patches = img_patches.reshape(batch_size, 3, max_patches, height, width)
        print(img_patches[0,:,20,200,200])
        sys.exit()

        for j in range(max_patches):
            cv2.imwrite(f"patches/{i:02d}_{j:04d}.png", torch.movedim(img_patches[0,:,i], (0,1,2), (2,0,1)).cpu().numpy()*255 )
            cv2.imwrite(f"patches/{i:02d}.png", torch.movedim(imgs[0,i], (0,1,2), (2,0,1)).cpu().numpy()*255 )

        #   #### visual
        #   # plot src patches
        #   r,c = 15,20
        #   ep = epipolar_grid[0,:,r,c,:].cpu().numpy()

        #   fig = plt.figure()
        #   ax = fig.add_subplot(111)
        #   ax.imshow(torch.movedim(imgs[0,i],(0,1,2),(2,0,1)).cpu().numpy())
        #   for x_i,y_i in ep:
        #       if x_i>=0 and y_i>=0:
        #           rect_i = Rectangle((x_i-(half_patch_size),y_i-(half_patch_size)), patch_size, patch_size, color='red', fc = 'none', lw = 0.5)
        #           ax.add_patch(rect_i)
        #   plt.savefig(f"src_img_{i}.png")
        #   plt.close()
        #   #### visual

    #   #### visual
    #   # plot ref point
    #   ref_pix = xy[0,r,c,:,0].cpu().numpy()
    #   plt.imshow(torch.movedim(imgs[0,0],(0,1,2),(2,0,1)).cpu().numpy())
    #   plt.plot(ref_pix[0].item(),ref_pix[1].item(),'ro')
    #   plt.savefig(f"ref_img.png")
    #   plt.close()
    #   #### visual

    sys.exit()

def essential_from_features(src_image_file: str, tgt_image_file: str, K: np.ndarray) -> np.ndarray:
    """Computes the essential matrix between two images using image features.

    Parameters:
        src_image_file: Input file for the source image.
        tgt_image_file: Input file for the target image.
        K: Intrinsics matrix of the two cameras (assumed to be constant between views).

    Returns:
        The essential matrix betweent the two image views.
    """
    src_image = cv2.imread(src_image_file)
    tgt_image = cv2.imread(tgt_image_file)

    # compute matching features
    (src_points, tgt_points) = match_features(src_image, tgt_image)

    # Compute fundamental matrix
    E, mask = cv2.findEssentialMat(src_points, tgt_points, K, method=cv2.RANSAC)

    return E

def fundamental_from_KP(K: np.ndarray, P_src: np.ndarray, P_tgt: np.ndarray) -> np.ndarray:
    """Computes the fundamental matrix between two images using camera parameters.

    Parameters:
        K: Intrinsics matrix of the two cameras (assumed to be constant between views).
        P_src: Extrinsics matrix for the source view.
        P_tgt: Extrinsics matrix for the target view.

    Returns:
        The fundamental matrix betweent the two cameras.
    """
    F_mats = []
    for i in range(K.shape[0]):
        R1 = P_src[i,0:3,0:3]
        t1 = P_src[i,0:3,3]
        R2 = P_tgt[i,0:3,0:3]
        t2 = P_tgt[i,0:3,3]

        t1aug = torch.tensor([t1[0], t1[1], t1[2], 1]).to(K)
        epi2 = torch.matmul(P_tgt[i],t1aug)
        epi2 = torch.matmul(K[i],epi2[0:3])

        R = torch.matmul(R2,torch.t(R1))
        t = t2 - torch.matmul(R,t1)
        K1inv = torch.linalg.inv(K[i])
        K2invT = torch.t(K1inv)
        tx = torch.tensor([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]).to(K)
        F = torch.matmul(K2invT,torch.matmul(tx,torch.matmul(R,K1inv)))
        F = F/(torch.max(F)+1e-10)
        F_mats.append(F)

    return torch.stack(F_mats, dim=0)

def _fundamental_from_KP(K: np.ndarray, P_src: np.ndarray, P_tgt: np.ndarray) -> np.ndarray:
    """Computes the fundamental matrix between two images using camera parameters.

    Parameters:
        K: Intrinsics matrix of the two cameras (assumed to be constant between views).
        P_src: Extrinsics matrix for the source view.
        P_tgt: Extrinsics matrix for the target view.

    Returns:
        The fundamental matrix betweent the two cameras.
    """
    R1 = P_src[0:3,0:3]
    t1 = P_src[0:3,3]
    R2 = P_tgt[0:3,0:3]
    t2 = P_tgt[0:3,3]

    t1aug = np.array([t1[0], t1[1], t1[2], 1])
    epi2 = np.matmul(P_tgt,t1aug)
    epi2 = np.matmul(K,epi2[0:3])

    R = np.matmul(R2,np.transpose(R1))
    t= t2- np.matmul(R,t1)
    K1inv = np.linalg.inv(K)
    K2invT = np.transpose(K1inv)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.matmul(K2invT,np.matmul(tx,np.matmul(R,K1inv)))
    F = F/np.amax(F)

    return F

def fundamental_from_features(src_image_file: str, tgt_image_file: str) -> np.ndarray:
    """Computes the fundamental matrix between two images using image features.

    Parameters:
        src_image_file: Input file for the source image.
        tgt_image_file: Input file for the target image.

    Returns:
        The fundamental matrix betweent the two image views.
    """
    src_image = cv2.imread(src_image_file)
    tgt_image = cv2.imread(tgt_image_file)

    # compute matching features
    (src_points, tgt_points) = match_features(src_image, tgt_image)
    
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(src_points,tgt_points,cv2.FM_8POINT)

    return F

def geometric_consistency_error(src_depth: np.ndarray, src_cam: np.ndarray, tgt_depth: np.ndarray, tgt_cam: np.ndarray) -> np.ndarray:
    """Computes the geometric consistency error between a source and target depth map.

    Parameters:
        src_depth: Depth map for the source view.
        src_cam: Camera parameters for the source depth map viewpoint.
        tgt_depth: Depth map for the target view.
        tgt_cam: Camera parameters for the target depth map viewpoint.

    Returns:
        The binary consistency mask encoding depth consensus between source and target depth maps.
    """
    height, width = src_depth.shape
    x_src, y_src = np.meshgrid(np.arange(0, width), np.arange(0, height))

    depth_reprojected, coords_reprojected, coords_tgt, projection_map = reproject(src_depth, src_cam, tgt_depth, tgt_cam)

    dist = np.sqrt((coords_reprojected[:,:,0] - x_src) ** 2 + (coords_reprojected[:,:,1] - y_src) ** 2)

    return dist, projection_map

def geometric_consistency_mask(src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P, pixel_th):
    """Computes the geometric consistency mask between a source and target depth map.

    Parameters:
        src_depth: Depth map for the source view.
        src_K: Intrinsic camera parameters for the source depth map viewpoint.
        src_P: Extrinsic camera parameters for the source depth map viewpoint.
        tgt_depth: Target depth map used for re-projection.
        tgt_K: Intrinsic camera parameters for the target depth map viewpoint.
        tgt_P: Extrinsic camera parameters for the target depth map viewpoint.
        pixel_th: Pixel re-projection threshold to determine matching depth estimates.

    Returns:
        The binary consistency mask encoding depth consensus between source and target depth maps.
    """
    batch_size, c, height, width = src_depth.shape
    depth_reprojected, coords_reprojected, coords_tgt = reproject(src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P)

    x_src, y_src = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing="xy")
    x_src = x_src.unsqueeze(0).repeat(batch_size, 1, 1).to(src_depth)
    y_src = y_src.unsqueeze(0).repeat(batch_size, 1, 1).to(src_depth)
    dist = torch.sqrt((coords_reprojected[:,:,:,0] - x_src) ** 2 + (coords_reprojected[:,:,:,1] - y_src) ** 2)

    mask = torch.where(dist < pixel_th, 1, 0)
    return mask

def _geometric_consistency_mask(src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P, pixel_th):
    """Computes the geometric consistency mask between a source and target depth map.

    Parameters:
        src_depth: Depth map for the source view.
        src_K: Intrinsic camera parameters for the source depth map viewpoint.
        src_P: Extrinsic camera parameters for the source depth map viewpoint.
        tgt_depth: Target depth map used for re-projection.
        tgt_K: Intrinsic camera parameters for the target depth map viewpoint.
        tgt_P: Extrinsic camera parameters for the target depth map viewpoint.
        pixel_th: Pixel re-projection threshold to determine matching depth estimates.

    Returns:
        The binary consistency mask encoding depth consensus between source and target depth maps.
    """
    height, width = src_depth.shape
    x_src, y_src = np.meshgrid(np.arange(0, width), np.arange(0, height))

    depth_reprojected, coords_reprojected, coords_tgt = _reproject(src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P)

    dist = np.sqrt((coords_reprojected[:,:,0] - x_src) ** 2 + (coords_reprojected[:,:,1] - y_src) ** 2)
    mask = np.where(dist < pixel_th, 1, 0)

    return mask

def get_uncovered_mask(data, output):
    hypos = output["hypos"]
    intervals = output["intervals"]
    _,H,W = data["target_depth"].shape
    levels = len(hypos)

    uncovered_masks = torch.zeros(levels,H,W)
    for l in range(levels):
        hypo = hypos[l].squeeze(1)
        batch_size, planes, h, w = hypo.shape
        interval = intervals[l].squeeze(1)
        target_depth = tvf.resize(data["target_depth"], [h,w]).unsqueeze(1)

        ### compute coverage
        diff = torch.abs(hypo - target_depth)
        min_interval = interval[:,0:1] * 0.5 # intervals are bin widths, divide by 2 for radius
        coverage = torch.clip(torch.where(diff <= min_interval, 1, 0).sum(dim=1, keepdim=True), 0, 1)
        uncovered = torch.clip(torch.where(coverage <= 0, 1, 0).sum(dim=1, keepdim=True), 0, 1)
        valid_targets = torch.where(target_depth > 0, 1, 0)
        uncovered *= valid_targets
        uncovered_masks[l] = (tvf.resize(uncovered.reshape(1,h,w), [H,W], interpolation=tvt.InterpolationMode.NEAREST)).squeeze(0)

    return uncovered_masks

def homography(src_image_file: str, tgt_image_file: str) -> np.ndarray:
    """Computes a homography transformation between two images using image features.

    Parameters:
        src_image_file: Input file for the source image.
        tgt_image_file: Input file for the target image.

    Returns:
        The homography matrix to warp the target image to the source image.
    """
    src_image = cv2.imread(src_image_file)
    tgt_image = cv2.imread(tgt_image_file)

    (height, width, _) = src_image.shape

    (src_points, tgt_points) = match_features(src_image, tgt_image)

    # Compute fundamental matrix
    H, mask = cv2.findHomography(tgt_points, src_points, method=cv2.RANSAC)

    return H

def homography_warp(cfg, features, level, ref_in, src_in, ref_ex, src_ex, depth_hypos, gwc_groups, va_net=None, vis_weights=None, aggregation="variance"):
    """Performs homography warping to create a Plane Sweeping Volume (PSV).
    Parameters:
        cfg: Configuration dictionary containing configuration parameters.
        features: Feature maps to be warped into a PSV.
        level: Current feature resolution level.
        ref_in: Reference view intrinsics matrix.
        src_in: Source view intrinsics matrices.
        ref_ex: Reference view extrinsics matrix.
        src_ex: Source view extrinsics matrices.
        depth_hypos: Depth hypotheses to use for homography warping.
        gwc_groups: Feature channel sizes used in group-wise correlation.
        va_net: Network used for visibility weighting.
        vis_weights: Pre-computed visibility weights.
        aggregation: Aggregation method to be used.

    Returns:
        The Plane Sweeping Volume computed via feature matching cost.
    """
    depth_hypos = depth_hypos.squeeze(1)
    _,planes,_,_ = depth_hypos.shape

    B,fCH,H,W = features[0][level].shape
    num_depth = depth_hypos.shape[1]
    nSrc = len(features)-1

    vis_weight_list = []
    ref_volume = features[0][level].unsqueeze(2).repeat(1,1,num_depth,1,1)

    if aggregation == "weighted_mean":
        cost_volume = None
    elif aggregation == "variance":
        cost_volume = torch.zeros((nSrc+1,B,fCH,planes,H,W)).to(features[0][level])
        cost_volume[0] = ref_volume

    reweight_sum = None
    for src in range(nSrc):
        with torch.no_grad():
            with autocast(enabled=False):
                src_proj = torch.matmul(src_in[:,src,:,:],src_ex[:,src,0:3,:])
                ref_proj = torch.matmul(ref_in,ref_ex[:,0:3,:])
                last = torch.tensor([[[0,0,0,1.0]]]).repeat(len(src_in),1,1).cuda()
                src_proj = torch.cat((src_proj,last),1)
                ref_proj = torch.cat((ref_proj,last),1)

                proj = torch.matmul(src_proj, torch.inverse(ref_proj))
                rot = proj[:, :3, :3]  # [B,3,3]
                trans = proj[:, :3, 3:4]  # [B,3,1]

                y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=ref_volume.device),
                                    torch.arange(0, W, dtype=torch.float32, device=ref_volume.device)],
                                    indexing='ij')
                y, x = y.contiguous(), x.contiguous()
                y, x = y.view(H * W), x.view(H * W)
                xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
                xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
                rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

                rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(B, 1, num_depth,H*W)  # [B, 3, Ndepth, H*W]
                proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
                proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
                proj_x_normalized = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1
                proj_y_normalized = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
                proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
                grid = proj_xy

        
        grid = grid.type(ref_volume.dtype)
        src_feature = features[src+1][level]
        warped_src_fea = F.grid_sample(src_feature, grid.view(B, num_depth * H, W, 2), mode='bilinear',
                                    padding_mode='zeros',align_corners=False)
        warped_src_fea = warped_src_fea.view(B, fCH, num_depth, H, W)


        if aggregation == "weighted_mean":
            ########## dot prod ##########
            two_view_cost_volume = groupwise_correlation(warped_src_fea, ref_volume, gwc_groups[level]) #B,C,D,H,W
            ## Estimate visability weight for init level
            if va_net is not None:
                B,C,D,H,W = warped_src_fea.shape
                reweight = va_net(two_view_cost_volume) #B, H, W
                vis_weight_list.append(reweight)
                reweight = reweight.unsqueeze(1).unsqueeze(2) #B, 1, 1, H, W
                two_view_cost_volume = reweight*two_view_cost_volume
            ## Use estimated visability weights for refine levels
            elif vis_weights is not None:
                reweight = vis_weights[src].unsqueeze(1)
                if reweight.shape[2] < two_view_cost_volume.shape[3]:
                    reweight = F.interpolate(reweight,scale_factor=2,mode='bilinear',align_corners=False)
                vis_weight_list.append(reweight.squeeze(1))
                reweight = reweight.unsqueeze(2)
                two_view_cost_volume = reweight*two_view_cost_volume
            if cost_volume == None:
                cost_volume = two_view_cost_volume
                reweight_sum = reweight
            else:
                cost_volume = cost_volume + two_view_cost_volume
                reweight_sum = reweight_sum + reweight

            if cfg["mode"]=="inference":
                del src_feature
                del two_view_cost_volume
                del warped_src_fea
                del reweight
                torch.cuda.empty_cache()
            ########### dot prod ##########
        elif aggregation == "variance":
            ######## var prod ##########
            ## Estimate visability weight for init level
            if va_net is not None:
                two_view_cost_volume = groupwise_correlation(warped_src_fea, ref_volume, gwc_groups[level]) #B,C,D,H,W
                B,C,D,H,W = warped_src_fea.shape
                reweight = va_net(two_view_cost_volume) #B, H, W
                vis_weight_list.append(reweight)
                reweight = reweight.unsqueeze(1).unsqueeze(2) #B, 1, 1, H, W
                warped_src_fea = reweight*warped_src_fea
            ## Use estimated visability weights for refine levels
            elif vis_weights is not None:
                reweight = vis_weights[src].unsqueeze(1)
                if reweight.shape[2] < cost_volume[src+1].shape[3]:
                    reweight = F.interpolate(reweight,scale_factor=2,mode='bilinear',align_corners=False)
                vis_weight_list.append(reweight.squeeze(1))
                reweight = reweight.unsqueeze(2)
                warped_src_fea = reweight*warped_src_fea

            cost_volume[src+1] = warped_src_fea

            if cfg["mode"]=="inference":
                del src_feature
                del warped_src_fea
                torch.cuda.empty_cache()
            ########## var prod ##########

    if aggregation == "weighted_mean":
        cost_volume = cost_volume/(reweight_sum+0.00001)
    elif aggregation == "variance":
        cost_volume = torch.var(cost_volume, dim=0)
        B,C,D,H,W = cost_volume.shape
        cost_volume = cost_volume.mean(dim=1, keepdim=True)

    return cost_volume, vis_weight_list


#   def homography_warp_coords(cfg, features, level, ref_in, src_in, ref_pose, src_pose, depth_hypos, coords, H, W, gwc_groups, va_net=None, vis_weights=None):
#       batch_size, c, h, w = features[0][level].shape
#       _, _, num_planes, _, _ = depth_hypos.shape
#       _, num_pixels, _ = coords.shape
#       num_src_views = len(features)-1
#   
#       K_ref = torch.zeros(batch_size, 4, 4).to(ref_in)
#       K_ref[:, :3,:3] = ref_in
#       K_ref[:, 3, 3] = 1
#   
#       K_src = torch.zeros(batch_size, num_src_views, 4, 4).to(ref_in)
#       for v in range(num_src_views):
#           K_src[:, v, :3, :3] = src_in[:,v]
#       K_src[:, :, 3, 3] = 1
#   
#       vis_weight_list = []
#       cost_volume = None
#       reweight_sum = None
#   
#       # build coordinates vector
#       depth_hypos = depth_hypos.reshape(batch_size, num_planes, num_pixels) # batch_size x num_planes x num_pixels
#       coords = torch.movedim(coords, (0,1,2), (0,2,1)).to(torch.float32) # batch_size x 2 x num_pixels
#       x_coords = coords[:,1,:]
#       y_coords = coords[:,0,:]
#       xyz = torch.stack((x_coords, y_coords, torch.ones_like(x_coords)), dim=1) # batch_size, 3 x num_pixels
#   
#       # sample reference features
#       x_normalized = ((x_coords / (W-1)) * 2) - 1
#       y_normalized = ((y_coords / (H-1)) * 2) - 1
#       xy = torch.stack((x_normalized, y_normalized), dim=-1)  # [B, num_pixels, 2]
#       ref_features = features[0][level]
#       ref_features = F.grid_sample(ref_features,
#                               xy.view(batch_size, 1, num_pixels, 2),
#                               mode='nearest',
#                               padding_mode='zeros',
#                               align_corners=False)
#       ref_features = ref_features.repeat(1,1,num_planes,1) # [B x C x num_pixels x num_planes]
#   
#       for src in range(num_src_views):
#           with torch.no_grad():
#               src_proj = torch.matmul(K_src[:,src],src_pose[:,src])
#               ref_proj = torch.matmul(K_ref,ref_pose)
#   
#               proj = torch.matmul(src_proj, torch.inverse(ref_proj))
#               rot = proj[:, :3, :3]  # [B,3,3]
#               trans = proj[:, :3, 3:4]  # [B,3,1]
#   
#               rot_xyz = torch.matmul(rot, xyz)  # [B, 3, num_pixels]
#               rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_planes, 1) * depth_hypos.view(batch_size, 1, num_planes, num_pixels)  # [B, 3, num_planes, num_pixels]
#               proj_xyz = rot_depth_xyz + trans.view(batch_size, 3, 1, 1)  # [B, 3, num_planes, num_pixels]
#               proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, num_planes, num_pixels]
#               proj_x_normalized = ((proj_xy[:, 0, :, :] / (W-1)) * 2) - 1
#               proj_y_normalized = ((proj_xy[:, 1, :, :] / (H-1)) * 2) - 1
#   
#               proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, num_planes, num_pixels, 2]
#               grid = proj_xy
#   
#           grid = grid.type(ref_features.dtype)
#           # Note: The shape of the input map is 4D: Batch x Channels x Height x Width.
#           #       The input coordinates must be in [x, y] format where x->width, y->height.
#           #       These coordinates must be normalized between [-1, 1].
#           src_features = features[src+1][level]
#           src_features = F.grid_sample(src_features,
#                                   grid.view(batch_size, num_planes, num_pixels, 2),
#                                   mode='bilinear',
#                                   padding_mode='zeros',
#                                   align_corners=False)
#           two_view_cost_volume = groupwise_correlation(src_features, ref_features, gwc_groups[level]) #B,C,num_planes,num_pixels
#           two_view_cost_volume = two_view_cost_volume.reshape(batch_size, gwc_groups[level], num_planes, H, W)
#   
#           # Estimate visability weight for init level
#           if va_net is not None:
#               reweight = va_net(two_view_cost_volume) #B, H, W
#               vis_weight_list.append(reweight)
#               reweight = reweight.unsqueeze(1).unsqueeze(2) #B, 1, 1, H, W
#               two_view_cost_volume = reweight*two_view_cost_volume
#   
#           # Use estimated visability weights for refine levels
#           elif vis_weights is not None:
#               reweight = vis_weights[src].unsqueeze(1)
#               if reweight.shape[2] < two_view_cost_volume.shape[3]:
#                   reweight = F.interpolate(reweight,scale_factor=2,mode='bilinear',align_corners=False)
#               vis_weight_list.append(reweight.squeeze(1))
#               reweight = reweight.unsqueeze(2)
#               two_view_cost_volume = reweight*two_view_cost_volume
#   
#           if cost_volume == None:
#               cost_volume = two_view_cost_volume
#               reweight_sum = reweight
#           else:
#               cost_volume = cost_volume + two_view_cost_volume
#               reweight_sum = reweight_sum + reweight
#   
#       cost_volume = cost_volume/(reweight_sum+1e-5)
#       return cost_volume, vis_weight_list


def match_features(src_image: np.ndarray, tgt_image: np.ndarray, max_features: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Computer matching ORB features between a pair of images.

    Args:
        src_image: The source image to compute and match features.
        tgt_image: The target image to compute and match features.
        max_features: The maximum number of features to retain.

    Returns:
        src_points: The set of matched point coordinates for the source image.
        tgt_points: The set of matched point coordinates for the target image.
    """
    src_image = cv2.cvtColor(src_image,cv2.COLOR_BGR2GRAY)
    tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2GRAY)
      
    orb = cv2.ORB_create(max_features)
      
    src_keypoints, src_descriptors = orb.detectAndCompute(src_image,None)
    tgt_keypoints, tgt_descriptors = orb.detectAndCompute(tgt_image,None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = list(matcher.match(src_descriptors, tgt_descriptors) )
    matches.sort(key = lambda x:x.distance)

    src_points = []
    tgt_points = []
    for i in range(8):
        m = matches[i]

        src_points.append(src_keypoints[m.queryIdx].pt)
        tgt_points.append(tgt_keypoints[m.trainIdx].pt)
    src_points  = np.asarray(src_points)
    tgt_points = np.asarray(tgt_points)

    return (src_points, tgt_points)

def plane_coords(K, P, depth_hypos, H, W):
    """Batched PyTorch version
    """
    batch_size,_,_ = K.shape
    num_planes = depth_hypos.shape[0]

    xyz = torch.movedim(torch.tensor([[0,0,1], [W-1,0,1], [0,H-1,1], [W-1,H-1,1]], dtype=torch.float32), 0, 1).to(P)
    xyz = xyz.reshape(1,3,4).repeat(batch_size, 1, 1)
    if K.shape[1]==3:
        K_44 = torch.zeros((batch_size, 4, 4)).to(P)
        K_44[:,:3,:3] = K[:,:3,:3]
        K_44[:,3,3] = 1
        K = K_44
    proj = K @ P
    inv_proj = torch.linalg.inv(proj)

    planes = torch.zeros(num_planes, 3, 4).to(inv_proj)
    for p in range(num_planes):
        planes[p] = (inv_proj[0,:3,:3] @ xyz) * depth_hypos[p]
        planes[p] += inv_proj[0,:3,3:4]

    return planes


def _plane_coords(K, P, near, far, H, W):
    """Numpy version
    """
    xyz = np.asarray([[0,0,1], [W-1,0,1], [0,H-1,1], [W-1,H-1,1]], dtype=np.float32).transpose()
    if K.shape[0]==3:
        K_44 = np.zeros((4, 4))
        K_44[:3,:3] = K[:3,:3]
        K_44[3,3] = 1
        K = K_44
    proj = K @ P

    near_plane = (np.linalg.inv(proj)[:3,:3] @ xyz) * near
    near_plane += np.linalg.inv(proj)[:3,3:4]
    far_plane = (np.linalg.inv(proj)[:3,:3] @ xyz) * far
    far_plane += np.linalg.inv(proj)[:3,3:4]

    return near_plane, far_plane

def points_from_depth(depth: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Creates a point array from a single depth map.

    Parameters:
        depth: Depth map to project to 3D.
        cam: Camera parameters for the given depth map viewpoint.

    Returns:
        An array of 3D points corresponding to the input depth map.
    """
    # project depth map to point cloud
    height, width = depth.shape
    x = np.linspace(0,width-1,num=width)
    y = np.linspace(0,height-1,num=height)
    x,y = np.meshgrid(x,y, indexing="xy")
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    xyz_cam = np.matmul(np.linalg.inv(cam[1,:3,:3]), np.vstack((x, y, np.ones_like(x))) * depth)
    xyz_world = np.matmul(np.linalg.inv(cam[0,:4,:4]), np.vstack((xyz_cam, np.ones_like(x))))[:3]
    points = xyz_world.transpose((1, 0))
    return points

def project_depth_map(depth: torch.Tensor, cam: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Projects a depth map into a list of 3D points

    Parameters:
        depth: Input depth map to project.
        cam: Camera parameters for input depth map.

    Returns:
        A float Tensor of 3D points corresponding to the projected depth values.
    """
    if (depth.shape[1] == 1):
        depth = depth.squeeze(1)

    batch_size, height, width = depth.shape
    cam_shape = cam.shape

    # get camera extrinsics and intrinsics
    P = cam[:,0,:,:]
    K = cam[:,1,:,:]
    K[:,3,:] = torch.tensor([0,0,0,1])

    # construct back-projection from invers matrices
    # separate into rotation and translation components
    bwd_projection = torch.matmul(torch.inverse(P), torch.inverse(K)).to(torch.float32)
    bwd_rotation = bwd_projection[:,:3,:3]
    bwd_translation = bwd_projection[:,:3,3:4]

    # build 2D homogeneous coordinates tensor: [B, 3, H*W]
    with torch.no_grad():
        row_span = torch.arange(0, height, dtype=torch.float32).cuda()
        col_span = torch.arange(0, width, dtype=torch.float32).cuda()
        r,c = torch.meshgrid(row_span, col_span, indexing="ij")
        r,c = r.contiguous(), c.contiguous()
        r,c = r.reshape(height*width), c.reshape(height*width)
        coords = torch.stack((c,r,torch.ones_like(c)))
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1)

    # compute 3D coordinates using the depth map: [B, H*W, 3]
    world_coords = torch.matmul(bwd_rotation, coords)
    depth = depth.reshape(batch_size, 1, -1)
    world_coords = world_coords * depth
    world_coords = world_coords + bwd_translation

    #TODO: make sure index select is differentiable
    #       (there is a backward function but need to find the code..)
    if (mask != None):
        world_coords = torch.index_select(world_coords, dim=2, index=non_zero_inds)
        world_coords = torch.movedim(world_coords, 1, 2)

    # reshape 3D coordinates back into 2D map: [B, H, W, 3]
    #   coords_map = world_coords.reshape(batch_size, height, width, 3)

    return world_coords

def project_renderer(renderer: o3d.visualization.rendering.OffscreenRenderer, K: np.ndarray, P: np.ndarray, width: float, height: float) -> np.ndarray:
    """Projects the scene in an Open3D Offscreen Renderer to the 2D image plane.

    Parameters:
        renderer: Geometric scene to be projected.
        K: Camera intrinsic parameters.
        P: Camera extrinsic parameters.
        width: Desired image width.
        height: Desired image height.

    Returns:
        The rendered image for the scene at the specified camera viewpoint.
    """
    # set up the renderer
    intrins = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
    renderer.setup_camera(intrins, P)

    # render image
    image = np.asarray(renderer.render_to_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def render_custom_values(points: np.ndarray, values: np.ndarray, image_shape: Tuple[int,int], cam: np.ndarray) -> np.ndarray:
    """Renders a point cloud into a 2D camera plane using custom values for each pixel.

    Parameters:
        points: List of 3D points to be rendered.
        values: List of values to be written in the rendered image.
        image_shape: Desired shape (height,width) of the rendered image.
        cam: Camera parameters for the image viewpoint.

    Returns:
        The rendered image for the list of points using the sepcified corresponding values.
    """
    points = points.tolist()
    values = list(values.astype(float))
    cam = cam.flatten().tolist()

    rendered_img = rd.render(list(image_shape), points, values, cam)

    return rendered_img

def _render_point_cloud(cloud: o3d.geometry.PointCloud, pose: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Renders a point cloud into a 2D image plane.

    Parameters:
        cloud: Point cloud to be rendered.
        pose: Camera extrinsic parameters for the image plane.
        K: Camera intrinsic parameters for the image plane.
        width: Desired width of the rendered image.
        height: Desired height of the rendered image.

    Returns:
        The rendered image for the point cloud at the specified camera viewpoint.
    """
    #   cmap = plt.get_cmap("hot_r")
    #   colors = cmap(dists)[:, :3]
    #   ply.colors = o3d.utility.Vector3dVector(colors)

    # set up the renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    render.scene.add_geometry("cloud", cloud, mat)
    render.scene.set_background(np.asarray([0,0,0,1])) #r,g,b,a
    intrins = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
    render.setup_camera(intrins, pose)

    # render image
    image = np.asarray(render.render_to_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    depth = np.asarray(render.render_to_depth_image(z_in_view_space=True))

    return image, depth

def render_point_cloud_single(cloud: o3d.geometry.PointCloud, pose: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Renders a point cloud into a 2D image plane.

    Parameters:
        cloud: Point cloud to be rendered.
        pose: Camera extrinsic parameters for the image plane.
        K: Camera intrinsic parameters for the image plane.
        width: Desired width of the rendered image.
        height: Desired height of the rendered image.

    Returns:
        The rendered image for the point cloud at the specified camera viewpoint.
    """
    #   cmap = plt.get_cmap("hot_r")
    #   colors = cmap(dists)[:, :3]
    #   ply.colors = o3d.utility.Vector3dVector(colors)

    # set up the renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    render.scene.add_geometry("cloud", cloud, mat)
    render.scene.set_background(np.asarray([0,0,0,1])) #r,g,b,a
    intrins = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
    render.setup_camera(intrins, pose)

    # render image
    image = np.asarray(render.render_to_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    depth = np.asarray(render.render_to_depth_image(z_in_view_space=True))

    return image, depth

def render_point_cloud(render, intrins, pose):
    """Renders a point cloud into a 2D image plane.

    Parameters:

    Returns:
    """
    render.setup_camera(intrins, pose)

    # render image
    image = np.asarray(render.render_to_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    depth = np.asarray(render.render_to_depth_image(z_in_view_space=True))

    return image, depth

def reproject(src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P):
    """Computes the re-projection depth values and pixel indices between two depth maps.

    This function takes as input two depth maps: 'src_depth' and 'tgt_depth'. The source
    depth map is first projected into the target camera plane using the source depth
    values and the camera parameters for both views. Using the projected pixel
    coordinates in the target view, the target depths are then re-projected back into
    the source camera plane (again with the camera parameters for both views). The
    information prouced from this process is often used to compute errors in
    re-projection between two depth maps, or similar operations.

    Parameters:
        src_depth: Source depth map to be projected.
        src_K: Intrinsic camera parameters for the source depth map viewpoint.
        src_P: Extrinsic camera parameters for the source depth map viewpoint.
        tgt_depth: Target depth map used for re-projection.
        tgt_K: Intrinsic camera parameters for the target depth map viewpoint.
        tgt_P: Extrinsic camera parameters for the target depth map viewpoint.

    Returns:
        depth_reprojected: The re-projected depth values for the source depth map.
        coords_reprojected: The re-projection coordinates for the source view.
        coords_tgt: The projected coordinates for the target view.
    """
    batch_size, c, height, width = src_depth.shape

    # back-project ref depths to 3D
    x_src, y_src = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing="xy")
    x_src = x_src.reshape(-1).unsqueeze(0).repeat(batch_size, 1).to(src_depth)
    y_src = y_src.reshape(-1).unsqueeze(0).repeat(batch_size, 1).to(src_depth)
    homog = torch.stack((x_src, y_src, torch.ones_like(x_src)), dim=1)
    xyz_src = torch.matmul(torch.linalg.inv(src_K), homog * src_depth.reshape(batch_size, 1, -1))

    # transform 3D points from ref to src coords
    homog_3d = torch.concatenate((xyz_src, torch.ones_like(x_src).unsqueeze(1)), dim=1)
    xyz_tgt = torch.matmul(torch.matmul(tgt_P, torch.linalg.inv(src_P)), homog_3d)[:,:3]

    # project src 3D points into pixel coords
    K_xyz_tgt = torch.matmul(tgt_K, xyz_tgt)
    xy_tgt = K_xyz_tgt[:,:2] / K_xyz_tgt[:,2:3]
    x_tgt = xy_tgt[:,0].reshape(batch_size, height, width).to(torch.float32)
    y_tgt = xy_tgt[:,1].reshape(batch_size, height, width).to(torch.float32)
    coords_tgt = torch.stack((x_tgt, y_tgt), dim=-1) # B x H x W x 2

    # sample the depth values from the src map at each pixel coord
    x_normalized = ((x_tgt / (width-1)) * 2) - 1
    y_normalized = ((y_tgt / (height-1)) * 2) - 1
    grid = torch.stack((x_normalized, y_normalized), dim=-1) # B x H x W x 2
    sampled_depth_tgt = F.grid_sample(
                                    tgt_depth,
                                    grid,
                                    mode="bilinear",
                                    padding_mode="zeros",
                                    align_corners=False)

    # back-project src depths to 3D
    homog = torch.concatenate((xy_tgt, torch.ones_like(x_src).unsqueeze(1)), dim=1)
    xyz_tgt = torch.matmul(torch.linalg.inv(tgt_K), homog * sampled_depth_tgt.reshape(batch_size, 1, -1))

    # transform 3D points from src to ref coords
    homog_3d = torch.concatenate((xyz_tgt, torch.ones_like(x_src).unsqueeze(1)), dim=1)
    xyz_reprojected = torch.matmul(torch.matmul(src_P, torch.linalg.inv(tgt_P)), homog_3d)[:,:3]

    # extract reprojected depth values
    depth_reprojected = xyz_reprojected[:,2].reshape(batch_size, height, width).to(torch.float32)

    # project ref 3D points into pixel coords
    K_xyz_reprojected = torch.matmul(src_K, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:,:2] / (K_xyz_reprojected[:,2:3] + 1e-7)
    x_reprojected = xy_reprojected[:,0].reshape(batch_size, height, width).to(torch.float32)
    y_reprojected = xy_reprojected[:,1].reshape(batch_size, height, width).to(torch.float32)

    coords_reprojected = torch.stack((x_reprojected, y_reprojected), dim=-1) # B x H x W x 2

    return depth_reprojected, coords_reprojected, coords_tgt

def _reproject(src_depth, src_K, src_P, tgt_depth, tgt_K, tgt_P):
    """Computes the re-projection depth values and pixel indices between two depth maps.

    This function takes as input two depth maps: 'src_depth' and 'tgt_depth'. The source
    depth map is first projected into the target camera plane using the source depth
    values and the camera parameters for both views. Using the projected pixel
    coordinates in the target view, the target depths are then re-projected back into
    the source camera plane (again with the camera parameters for both views). The
    information prouced from this process is often used to compute errors in
    re-projection between two depth maps, or similar operations.

    Parameters:
        src_depth: Source depth map to be projected.
        src_K: Intrinsic camera parameters for the source depth map viewpoint.
        src_P: Extrinsic camera parameters for the source depth map viewpoint.
        tgt_depth: Target depth map used for re-projection.
        tgt_K: Intrinsic camera parameters for the target depth map viewpoint.
        tgt_P: Extrinsic camera parameters for the target depth map viewpoint.

    Returns:
        depth_reprojected: The re-projected depth values for the source depth map.
        coords_reprojected: The re-projection coordinates for the source view.
        coords_tgt: The projected coordinates for the target view.
    """
    height, width = src_depth.shape

    # back-project ref depths to 3D
    x_src, y_src = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_src, y_src = x_src.reshape([-1]), y_src.reshape([-1])
    xyz_src = np.matmul(np.linalg.inv(src_K),
                        np.vstack((x_src, y_src, np.ones_like(x_src))) * src_depth.reshape([-1]))

    # transform 3D points from ref to src coords
    xyz_tgt = np.matmul(np.matmul(tgt_P, np.linalg.inv(src_P)),
                        np.vstack((xyz_src, np.ones_like(x_src))))[:3]

    # project src 3D points into pixel coords
    K_xyz_tgt = np.matmul(tgt_K, xyz_tgt)
    xy_tgt = K_xyz_tgt[:2] / K_xyz_tgt[2:3]
    x_tgt = xy_tgt[0].reshape([height, width]).astype(np.float32)
    y_tgt = xy_tgt[1].reshape([height, width]).astype(np.float32)

    # sample the depth values from the src map at each pixel coord
    sampled_depth_tgt = cv2.remap(tgt_depth, x_tgt, y_tgt, interpolation=cv2.INTER_LINEAR)

    # back-project src depths to 3D
    xyz_tgt = np.matmul(np.linalg.inv(tgt_K),
                        np.vstack((xy_tgt, np.ones_like(x_src))) * sampled_depth_tgt.reshape([-1]))

    # transform 3D points from src to ref coords
    xyz_reprojected = np.matmul(np.matmul(src_P, np.linalg.inv(tgt_P)),
                                np.vstack((xyz_tgt, np.ones_like(x_src))))[:3]

    # extract reprojected depth values
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)

    # project ref 3D points into pixel coords
    K_xyz_reprojected = np.matmul(src_K, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-7)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    coords_reprojected = np.dstack((x_reprojected, y_reprojected))
    coords_tgt = np.dstack((x_tgt, y_tgt))

    return depth_reprojected, coords_reprojected, coords_tgt

def sample_volume(volume, z_vals, coords, H, W, near_depth, far_depth, inv_depth):
    """
    Parameters:

    Returns:
    """
    N, M = z_vals.shape
    batch_size, c, _, _, _ = volume.shape

    z_vals = z_vals.reshape(N,M,1) # N x M x 1
    if inv_depth:
        z_vals = 1/z_vals
        near_depth = 1/near_depth
        far_depth = 1/far_depth
    coords = coords.reshape(N,1,2).repeat(1,M,1) # N x M x 2
    x_coords = coords[:,:,1:2]
    y_coords = coords[:,:,0:1]
    points = torch.cat([x_coords, y_coords, z_vals], dim=-1) # N x M x 3
    points = torch.reshape(points, [-1, 3]) # N*M x 3

    # define coordinates bounds
    min_coord = torch.tensor([0,0,near_depth]).to(points)
    max_coord = torch.tensor([W-1,H-1,far_depth]).to(points)
    min_coord = min_coord.reshape(1,3).repeat(N*M,1)
    max_coord = max_coord.reshape(1,3).repeat(N*M,1) 

    # normalize points
    norm_points = (points - min_coord) / (max_coord - min_coord)
    norm_points = norm_points.unsqueeze(0).repeat(batch_size, 1, 1)
    norm_points = (norm_points * 2) - 1

    # Note: The shape of the input volume is 5D: Batch x Channels x Depth x Height x Width.
    #       The input coordinates must be in [x, y, z] format where x->width, y->height, z->depth.
    #       These coordinates must be normalized between [-1, 1].
    features = F.grid_sample(volume,
                            norm_points.view(batch_size, N*M, 1, 1, 3),
                            mode='bilinear',
                            padding_mode='zeros',
                            align_corners=True)
    features = torch.movedim(features.reshape(c, N*M), 0, 1) # N*M x c

    return features

def soft_hypothesis(data, target_hypo, focal_length, min_hypo, max_hypo, M, delta_in=1):
    """
    Parameters:

    Returns:
    """
    B, _, D, H, W = target_hypo.shape
    rand_match_offset = torch.rand(B,1,M,H,W).to(target_hypo)
    near, far = Z_from_disp(target_hypo, data["baseline"], focal_length, delta=delta_in)
    target_range = torch.abs(far - near).repeat(1,1,M,1,1)

    target_samples = (rand_match_offset * target_range) + near
    mask = torch.where(target_hypo <= 0, 0.0, 1.0).repeat(1,1,M,1,1)
    matching_hypos = torch.clip(target_samples, min_hypo, max_hypo) * mask

    return matching_hypos

def _soft_hypothesis(data, target_hypo, focal_length, min_hypo, max_hypo, M=1, delta_in=1):
    """
    Parameters:

    Returns:
    """
    B, _, D, H, W = target_hypo.shape
    rand_match_offset = torch.rand(B,1,M,H,W).to(target_hypo)
    rand_near_offset = torch.rand(B,1,M,H,W).to(target_hypo)
    rand_far_offset = torch.rand(B,1,M,H,W).to(target_hypo)

    near, far = Z_from_disp(target_hypo, data["baseline"], focal_length, delta=delta_in)
    target_range = torch.abs(far - near).repeat(1,1,M,1,1)
    near_range = torch.abs(near - min_hypo).repeat(1,1,M,1,1)
    far_range = torch.abs(max_hypo - far).repeat(1,1,M,1,1)

    target_samples = (rand_match_offset * target_range) + near
    near_samples = (rand_near_offset * near_range) + min_hypo
    far_samples = (rand_far_offset * far_range) + far
    samples = torch.cat([target_samples,near_samples,far_samples], dim=1)
    samples = samples.reshape(B,-1,H,W).unsqueeze(1) # [B, 1, M*3, H, W]

    mask = torch.where(target_hypo <= 0, 0.0, 1.0).repeat(1,1,M*3,1,1)
    hypos = torch.clip(samples, min_hypo, max_hypo) * mask

    return hypos

def resoluton_based_hypothesis(data, target_hypo, level, focal_length, min_hypo, max_hypo, delta_in=1):
    """
    Parameters:

    Returns:
    """
    B, _, D, H, W = target_hypo.shape
    rand_match_offset = torch.rand(B,1,M,H,W).to(target_hypo)
    rand_near_offset = torch.rand(B,1,M,H,W).to(target_hypo)
    rand_far_offset = torch.rand(B,1,M,H,W).to(target_hypo)

    near, far = Z_from_disp(target_hypo, data["baseline"], focal_length, delta=delta_in)
    target_range = torch.abs(far - near).repeat(1,1,M,1,1)
    near_range = torch.abs(near - min_hypo).repeat(1,1,M,1,1)
    far_range = torch.abs(max_hypo - far).repeat(1,1,M,1,1)

    target_samples = (rand_match_offset * target_range) + near
    near_samples = (rand_near_offset * near_range) + min_hypo
    far_samples = (rand_far_offset * far_range) + far
    samples = torch.cat([target_samples,near_samples,far_samples], dim=1)
    samples = samples.reshape(B,-1,H,W).unsqueeze(1) # [B, 1, M*3, H, W]

    mask = torch.where(target_hypo <= 0, 0.0, 1.0).repeat(1,1,M*3,1,1)
    hypos = torch.clip(samples, min_hypo, max_hypo) * mask

    return hypos

def visibility(depths, K, Ps, vis_th, levels=4):
    """
    Parameters:

    Returns:
    """
    batch_size, views, c, H, W = depths.shape

    K_pyr = intrinsic_pyramid(K, levels)

    vis_maps = []
    vis_masks = []
    for l in range(levels):
        resized_depths = tvf.resize(depths[:,:,0], [int(H/(2**l)), int(W/(2**l))]).unsqueeze(2)
        batch_size, views, c, height, width = resized_depths.shape
        vis_map = torch.where(resized_depths[:,0] > 0.0, 1, 0)

        for i in range(1, views):
            mask = geometric_consistency_mask(resized_depths[:,0], K_pyr[:,l], Ps[:,0], resized_depths[:,i], K_pyr[:,l], Ps[:,i], pixel_th=0.5)
            vis_map += mask.unsqueeze(1)
        vis_map = vis_map.to(torch.float32)

        vis_maps.append(vis_map)
        vis_masks.append(torch.where(vis_map >= vis_th, 1, 0))
    return vis_maps, vis_masks

def _visibility(depths, K, Ps, vis_th=None):
    """
    Parameters:

    Returns:
    """
    views, height, width = depths.shape
    vis_map = np.where(depths[0] > 0.0, 1, 0)

    for i in range(1, views):
        mask = _geometric_consistency_mask(depths[0], K, Ps[0], depths[i], K, Ps[i], pixel_th=0.75)
        vis_map += mask
    vis_map = vis_map.astype(np.float32)

    if vis_th != None:
        vis_map = np.where(vis_map >= vis_th, 1, 0)

    return vis_map

def visibility_mask(src_depth: np.ndarray, src_cam: np.ndarray, depth_files: List[str], cam_files: List[str], src_ind: int = -1, pixel_th: float = 0.1) -> np.ndarray:
    """Computes a visibility mask between a provided source depth map and list of target depth maps.

    Parameters:
        src_depth: Depth map for the source view.
        src_cam: Camera parameters for the source depth map viewpoint.
        depth_files: List of target depth maps.
        cam_files: List of corresponding target camera parameters for each targte depth map viewpoint.
        src_ind: Index into 'depth_files' corresponding to the source depth map (if included in the list).
        pixel_th: Pixel re-projection threshold to determine matching depth estimates.

    Returns:
        The visibility mask for the source view.
    """
    height, width = src_depth.shape
    vis_map = np.not_equal(src_depth, 0.0).astype(np.double)

    for i in range(len(depth_files)):
        if (i==src_ind):
            continue

        # get files
        sdf = depth_files[i]
        scf = cam_files[i]

        # load data
        tgt_depth = read_pfm(sdf)
        tgt_cam = read_single_cam_sfm(scf,'r')

        mask = geometric_consistency_mask(src_depth, src_cam, tgt_depth, tgt_cam, pixel_th)
        vis_map += mask

    return vis_map.astype(np.float32)

def uniform_hypothesis(cfg, ref_in,src_in,ref_ex,src_ex,depth_min, depth_max, img_height, img_width, nhypothesis_init, inv_depth=False):
    """
    Parameters:

    Returns:
    """
    batchSize = ref_in.shape[0]
    depth_range = depth_max-depth_min

    depth_hypos = torch.zeros((batchSize,nhypothesis_init),device=ref_in.device)
    for b in range(0,batchSize):
        if inv_depth:
            depth_hypos[b] = 1/(torch.linspace(1/depth_min,1/depth_max,steps=nhypothesis_init,device=ref_in.device))
        else:
            depth_hypos[b] = torch.linspace(depth_min, depth_max, steps=nhypothesis_init,device=ref_in.device)
    depth_hypos = depth_hypos.unsqueeze(2).unsqueeze(3).repeat(1,1,img_height,img_width)

    # Make coordinate for depth hypothesis, to be used by sparse convolution.
    depth_hypo_coords = torch.zeros((batchSize,nhypothesis_init),device=ref_in.device)
    for b in range(0,batchSize):
        depth_hypo_coords[b] = torch.linspace(0,nhypothesis_init-1,steps=nhypothesis_init,device=ref_in.device)
    depth_hypo_coords = depth_hypo_coords.unsqueeze(2).unsqueeze(3).repeat(1,1,img_height,img_width)

    # Calculate hypothesis interval
    hypo_intervals = depth_hypos[:,1:]-depth_hypos[:,:-1]
    hypo_intervals = torch.cat((hypo_intervals,hypo_intervals[:,-1].unsqueeze(1)),dim=1)

    return depth_hypos.unsqueeze(1), depth_hypo_coords.unsqueeze(1), hypo_intervals.unsqueeze(1)
