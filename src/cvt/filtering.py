# cvt/filtering.py

"""A suite of common filtering utilities.

This module includes several functions for filtering depth maps.
"""

import numpy as np
import torch
import os
from typing import Tuple
from tqdm import tqdm
import cv2
from PIL import Image
from plyfile import PlyData, PlyElement

from cvt.geometry import reproject
from cvt.io import read_cluster_list, write_pfm, read_pfm

def conf_filter(depth_map: torch.Tensor, conf_map: torch.Tensor, device: str = 'cuda:0', min_conf: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
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

def geometric_filter(src_depth: np.ndarray, src_cam: np.ndarray, tgt_depth: np.ndarray, tgt_cam: np.ndarray, pix_th: float=1.0, depth_th: float=0.01) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Computes a geometric filter based off of pixel and depth reprojection error.

    Parameters:
        src_depth:
        src_cam:
        tgt_depth:
        tgt_depth:
        pix_th:
        depth_th:

    Returns:
        mask:
        depth_reprojected:
        coords_tgt:
    """
    width, height = src_depth.shape[1], src_depth.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, coords_reprojected, coords_tgt = reproject(src_depth, src_cam, tgt_depth, tgt_cam)
    
    # measure pixel difference
    dist = np.sqrt((coords_reprojected[:,:,0] - x_ref) ** 2 + (coords_reprojected[:,:,1] - y_ref) ** 2)

    # measure depth difference
    depth_diff = np.abs(depth_reprojected - src_depth)
    relative_depth_diff = depth_diff / src_depth

    # compute mask
    mask = np.logical_and(dist < pix_th, relative_depth_diff < depth_th)

    # apply mask to depth map
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, coords_tgt

def topk_filter(depth_map: np.ndarray, conf_map:np.ndarray, device: str='cuda:0', percent: float=0.3) -> Tuple[np.ndarray,np.ndarray]:
    """Filters a map by the top percentage of confidence values.

    Parameters:
        depth_map:
        conf_map:
        device:
        percent:

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

def topk_strict_filter(depth_map: np.ndarray, filter_prob: np.ndarray, device: str = 'cuda:0', percent: float = 0.3) -> Tuple[np.ndarray,np.ndarray]:
    """Filters a map by the top percentage of confidence values.

    Parameters:
        depth_map:
        filter_prob:
        device:
        percent:

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


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    all_views = list(range(0,49))
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) == 0:
                continue
            data.append((ref_view, src_views))
    return data

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-7)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, pix_th=1.0):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = dist < pix_th
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src

def consensus_filter(cfg, est_depth_path, est_conf_path, rgb_path, output_path, dataset, scene):
    K = dataset.K[scene]
    poses = dataset.get_all_poses(scene)
    pix_th = cfg["point_cloud"]["pix_th"]
    prob_th = cfg["point_cloud"]["prob_th"]
    num_consistent = cfg["point_cloud"]["num_consistent"]

    vertexs = []
    vertex_colors = []

    clusters = read_cluster_list(dataset.get_cluster_file(scene))
    nviews = len(clusters)

    out_mask_path = os.path.join(output_path, "masks")
    out_geo_mask_path = os.path.join(out_mask_path, "geometric")
    out_conf_mask_path = os.path.join(out_mask_path, "confidence")
    os.makedirs(out_mask_path, exist_ok=True)
    os.makedirs(out_geo_mask_path, exist_ok=True)
    os.makedirs(out_conf_mask_path, exist_ok=True)

    # for each reference view and the corresponding source views
    for (ref_frame, src_frames) in tqdm(clusters, desc="Building point cloud", unit="views"):
        ref_image = cv2.imread(os.path.join(rgb_path, f"{ref_frame:08d}.png"))
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB ) 
        ref_depth_est = read_pfm(os.path.join(est_depth_path, f"{ref_frame:08d}.pfm"))
        ref_pose = poses[ref_frame]
        if prob_th == 0.0:
            conf_mask = (np.ones(ref_depth_est.shape, dtype=np.int64) == 1)
        else:
            confidence = read_pfm(os.path.join(est_conf_path, f"{ref_frame:08d}.pfm"))
            conf_mask = confidence > prob_th
            if conf_mask.shape[0] != ref_depth_est.shape[0] or conf_mask.shape[1] != ref_depth_est.shape[1]:
                conf_mask_t = torch.tensor(conf_mask, dtype=torch.float32)
                conf_mask = torch.squeeze(
                    F.interpolate(
                        torch.unsqueeze(torch.unsqueeze(conf_mask_t, 0), 0), 
                        [ref_depth_est.shape[0], ref_depth_est.shape[1]], mode="nearest")).numpy() == 1.0

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        ### Geometric Mask ###
        geo_mask_sum = 0
        for src_frame in src_frames:
            src_pose = poses[src_frame]
            src_depth_est = read_pfm(os.path.join(est_depth_path, f"{src_frame:08d}.pfm"))

            # compute geometric mask
            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, K, ref_pose, src_depth_est, K, src_pose, pix_th=pix_th)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= num_consistent
        final_mask = np.logical_and(conf_mask, geo_mask)

        save_mask(os.path.join(out_conf_mask_path, f"{ref_frame:08d}_conf.png"), conf_mask)
        save_mask(os.path.join(out_geo_mask_path, f"{ref_frame:08d}_geo.png"), geo_mask)
        save_mask(os.path.join(out_mask_path, f"{ref_frame:08d}.png"), final_mask)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask

        # use either average or reference depth estimates
        #x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        x, y, depth = x[valid_points], y[valid_points], ref_depth_est[valid_points]

        color = ref_image[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(K),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_pose),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color[:,:3]).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')

    # save point cloud to scene-level points path
    ply_file_scene = os.path.join(output_path, f"{scene}.ply")
    PlyData([el]).write(ply_file_scene)


def reprojection_error(cfg, est_depth_path, rgb_path, out_error_path, dataset, scene, max_views=10):
    K = dataset.K[scene]
    poses = dataset.get_all_poses(scene)

    clusters = read_cluster_list(dataset.get_cluster_file(scene))
    nviews = len(clusters)

    # for each reference view and the corresponding source views
    for (ref_frame, src_frames) in tqdm(clusters, desc="Computing reprojection error", unit="views"):
        ref_image = cv2.imread(os.path.join(rgb_path, f"{ref_frame:08d}.png"))
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB ) 
        ref_depth_est = read_pfm(os.path.join(est_depth_path, f"{ref_frame:08d}.pfm"))
        height,width = ref_depth_est.shape
        ref_pose = poses[ref_frame]

        ### Geometric Mask ###
        errors = []
        for v,src_frame in enumerate(src_frames):
            src_pose = poses[src_frame]
            src_depth_est = read_pfm(os.path.join(est_depth_path, f"{src_frame:08d}.pfm"))

            # compute geometric mask
            width, height = ref_depth_est.shape[1], ref_depth_est.shape[0]
            x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
            _, x2d_reprojected, y2d_reprojected, _, _ = reproject_with_depth(
                                                            ref_depth_est,
                                                            K,
                                                            ref_pose,
                                                            src_depth_est,
                                                            K,
                                                            src_pose)

            errors.append((np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)).astype(np.float32))


        errors = np.asarray(errors)
        errors = np.sort(errors,axis=0)
        np.save(os.path.join(out_error_path, f"{ref_frame:08d}.npy"), errors)

        errors = errors[:max_views]

        #for v in range(max_views):
        #    write_pfm(os.path.join(out_error_path, f"{ref_frame:08d}_{v}.pfm"), errors[v])
        #    # visualize clipped version
        #    e = np.clip(errors[v], 0, 2)
        #    cv2.imwrite(os.path.join(out_error_path, f"{ref_frame:08d}_{v}.png"), (e-e.min())/(e.max()-e.min())*255)

        error_sum = errors.sum(axis=0)
        write_pfm(os.path.join(out_error_path, f"{ref_frame:08d}_sum.pfm"), error_sum)

        # visualize clipped version
        e = np.clip(error_sum, 0, 20)
        cv2.imwrite(os.path.join(out_error_path, f"{ref_frame:08d}_sum.png"), (e-e.min())/(e.max()-e.min())*255)

def laplacians(cfg, paths):

    # get depth map filenames
    est_depth_files = os.listdir(paths["depth"])
    est_depth_files = [edf for edf in est_depth_files if edf[-3:] == "pfm" ]
    est_depth_files.sort()

    for i, edf in enumerate(est_depth_files):
        ref_ind = int(edf[:8])
        image = cv2.imread(os.path.join(paths["rgb"], f"{ref_ind:08d}.png"))
        image = torch.movedim(torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (0,1,2),(1,2,0))
        c,height,width = image.shape

        image = image.reshape(1,c,height,width).to(torch.float32)
        est_depth = torch.tensor(read_pfm(os.path.join(paths["depth"], edf))).reshape(1,1,height,width).to(torch.float32)

        image_laplacian = laplacian_pyramid(image)
        est_depth_laplacian = laplacian_pyramid(est_depth)

        img_lap = image_laplacian[0,0].detach().cpu().numpy()
        img_lap_filename = os.path.join(paths["laplacian"], f"{ref_ind:08d}_image.pfm")
        write_pfm(img_lap_filename, img_lap)
        depth_lap = est_depth_laplacian[0,0].detach().cpu().numpy()
        depth_lap_filename = os.path.join(paths["laplacian"], f"{ref_ind:08d}_depth.pfm")
        write_pfm(depth_lap_filename, depth_lap)
