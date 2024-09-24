# cvt/metrics.py

"""Module including routines computing metrics.

This module includes the following functions:

- `abs_error(est_depth, gt_depth)` - Computes the absolute error between an estimated and groun-truth depth map.
- `accuracy_eval(est_ply, gt_ply, mask_th, est_filt=None, gt_filt=None)` - Computes the accuracy of an estimated point cloud against the provided ground-truth.
- `completeness_eval(est_ply, gt_ply, mask_th=20.0, est_filt=None, gt_filt=None)` - Computes the completeness of an estimated point cloud against the provided ground-truth.
- `filter_outlier_points(est_ply, gt_ply, outlier_th)` - Filters out points from an estimated point cloud that are farther than some threshold to the ground-truth point cloud.
"""

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from typing import Tuple, Optional
import torch

def MAE(estimate, target, reduction_dims, mask=None, relative=False):
    assert(estimate.shape==target.shape)
    error = estimate - target
    if relative:
        error /= target
    error = torch.abs(error)

    if mask != None:
        assert(error.shape==mask.shape)
        error *= mask
        error = (error.sum(dim=reduction_dims) / (mask.sum(dim=reduction_dims)+1e-10)).sum()
    else:
        error =  error.mean()
    return error

def RMSE(estimate, target, mask=None, relative=False):
    assert(estimate.shape==target.shape)
    error = estimate - target
    if relative:
        error /= target
    error = torch.square(error)

    assert(error.shape==mask.shape)
    error *= mask

    reduction_dims = tuple(range(1,len(target.shape)))
    error = (error.sum(dim=reduction_dims) / (mask.sum(dim=reduction_dims)+1e-10)).sum()
    return torch.sqrt(error)

def abs_error(est_depth: np.ndarray, gt_depth: np.ndarray) -> np.ndarray:
    """Computes the absolute error between an estimated and groun-truth depth map.

    Parameters:
        est_depth: Estimated depth map.
        gt_depth: Ground-truth depth map.

    Returns:
       The absolute error map for the estimated depth map. 
    """
    signed_error = est_depth - gt_depth

    # compute gt mask and number of valid pixels
    gt_mask = np.not_equal(gt_depth, 0.0).astype(np.double)
    error = np.abs(signed_error) * gt_mask

    return error

def accuracy_eval(
        est_ply: np.ndarray,
        gt_ply: np.ndarray,
        mask_th: float,
        est_filt: Optional[np.ndarray] = None,
        gt_filt: Optional[np.ndarray] = None) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    """Computes the accuracy of an estimated point cloud against the provided ground-truth.

    Parameters:
        est_ply: Estimated point cloud to be evaluated.
        gt_ply: Ground-truth point cloud.
        mask_th: Masking threshold used to remove points from the evaluation farther 
                    than a specified distance value.
        est_filt: Optional filter to remove unwanted point from the estimated 
                    point cloud in the evaluation
        gt_filt: Optional filter to remove unwanted point from the ground-truth
                    point cloud in the evaluation

    Returns:
        valid_est_ply: Point cloud containing all the valid evaluation points after filtering.
        dists_est: Estimated distances of all valid points in the estimated point cloud 
                    to the closest point in the ground-truth point cloud.
        colors_est: Estimated colors for the points in the valid point cloud.
    """
    # distance from est to gt
    dists_est = np.asarray(est_ply.compute_point_cloud_distance(gt_ply))

    # extract valid indices
    valid_inds_est = set(np.where(est_filt == 1)[0])
    valid_dists = set(np.where(dists_est <= mask_th)[0])
    valid_inds_est.intersection_update(valid_dists)
    valid_inds_est = np.asarray(list(valid_inds_est))

    # get distances and colors at valid indices
    valid_est_ply = est_ply.select_by_index(valid_inds_est)
    dists_est = dists_est[valid_inds_est]
    colors_est = np.asarray(est_ply.colors)[valid_inds_est]

    return valid_est_ply, dists_est, colors_est


def completeness_eval(
        est_ply: o3d.geometry.PointCloud,
        gt_ply: o3d.geometry.PointCloud,
        mask_th: float = 20.0,
        est_filt: Optional[np.ndarray] = None,
        gt_filt: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the completeness of an estimated point cloud against the provided ground-truth.

    Parameters:
        est_ply: Estimated point cloud to be evaluated.
        gt_ply: Ground-truth point cloud.
        mask_th: Masking threshold used to remove points from the evaluation farther 
                    than a specified distance value.
        est_filt: Optional filter to remove unwanted point from the estimated 
                    point cloud in the evaluation
        gt_filt: Optional filter to remove unwanted point from the ground-truth
                    point cloud in the evaluation

    Returns:
        ply_points: Point cloud vertices containing all the valid evaluation points after filtering.
        dists: Distances of all valid points in the ground-truth point cloud 
                    to the closest point in the estimated point cloud.
        colors: Colors for the points in the valid point cloud.
    """
    # build KD-Tree of estimated point cloud for querying
    tree = KDTree(np.asarray(est_ply.points), leaf_size=40)
    (dists, inds) = tree.query(np.asarray(gt_ply.points), k=1)

    # extract valid indices
    valid_inds = set(np.where(gt_filt == 1)[0])
    valid_inds.intersection_update(set(np.where(dists <= mask_th)[0]))
    valid_inds = np.asarray(list(valid_inds))

    dists = dists[valid_inds]
    inds = inds[valid_inds]
    ply_points = np.asarray(est_ply.points)[inds]
    colors = np.asarray(est_ply.colors)[inds]

    return ply_points, dists, colors


def filter_outlier_points(est_ply: o3d.geometry.PointCloud, gt_ply: o3d.geometry.PointCloud, outlier_th: float) -> o3d.geometry.PointCloud:
    """Filters out points from an estimated point cloud that are farther than some threshold to the ground-truth point cloud.

    Parameters:
        est_ply: Estimated point cloud to filter.
        gt_ply: Ground-truth point cloud for reference.
        outlier_th: Distance threshold used for filtering.

    Returns:
        The filtered point cloud.
    """
    dists_est = np.asarray(est_ply.compute_point_cloud_distance(gt_ply))
    valid_dists = np.where(dists_est <= outlier_th)[0]
    return est_ply.select_by_index(valid_dists)



#   def roc_curve(in_depth, out_depth, in_conf, out_conf, gt_depth, path, stats_file, view_num, scene, device):
#       if (scene == "Barn"):
#           th = 0.01
#       elif(scene == "Caterpillar"):
#           th = 0.005
#       elif(scene == "Church"):
#           th = 0.025
#       elif(scene == "Courthouse"):
#           th = 0.025
#       elif (scene =="Ignatius"):
#           th = 0.003
#       elif(scene == "Meetingroom"):
#           th = 0.01
#       elif(scene == "Truck"):
#           th = 0.005
#       elif(scene == "none"): # DTU
#           th = 2
#       else:
#           th = 0.02
#   
#       height, width = in_depth.shape
#   
#       valid_map = torch.ne(gt_depth, 0.0).to(torch.float32).to(device)
#       valid_count = torch.sum(valid_map)+1e-7
#   
#       # flatten to 1D tensor
#       in_depth = torch.flatten(in_depth)
#       in_conf = torch.flatten(in_conf)
#       out_depth = torch.flatten(out_depth)
#       out_conf = torch.flatten(out_conf)
#       gt_depth = torch.flatten(gt_depth)
#   
#       ##### INPUT #####
#       # sort all tensors by confidence value
#       (in_conf,indices) = in_conf.sort(descending=True)
#       in_depth = torch.gather(in_depth, dim=0, index=indices)
#       in_gt_depth = torch.gather(gt_depth, dim=0, index=indices)
#       # pull only gt values
#       in_gt_indices = torch.nonzero(in_gt_depth).flatten()
#       in_depth = torch.index_select(in_depth, dim=0, index=in_gt_indices)
#       in_gt_depth = torch.index_select(in_gt_depth, dim=0, index=in_gt_indices)
#   
#       # sort orcale curves by error
#       in_oracle = torch.abs(in_depth-in_gt_depth)
#       (in_oracle,indices) = in_oracle.sort(descending=False)
#       in_oracle_gt = torch.gather(in_gt_depth, dim=0, index=indices)
#       # pull only gt values
#       in_oracle_indices = torch.nonzero(in_oracle_gt).flatten()
#       in_oracle = torch.index_select(in_oracle, dim=0, index=in_oracle_indices)
#   
#       ##### OUTPUT #####
#       # sort all tensors by confidence value
#       (out_conf,indices) = out_conf.sort(descending=True)
#       out_depth = torch.gather(out_depth, dim=0, index=indices)
#       out_gt_depth = torch.gather(gt_depth, dim=0, index=indices)
#       # pull only gt values
#       out_gt_indices = torch.nonzero(out_gt_depth).flatten()
#       out_depth = torch.index_select(out_depth, dim=0, index=out_gt_indices)
#       out_gt_depth = torch.index_select(out_gt_depth, dim=0, index=out_gt_indices)
#   
#       out_oracle = torch.abs(out_depth-out_gt_depth)
#       (out_oracle,indices) = out_oracle.sort(descending=False)
#       out_oracle_gt = torch.gather(out_gt_depth, dim=0, index=indices)
#       # pull only gt values
#       out_oracle_indices = torch.nonzero(out_oracle_gt).flatten()
#       out_oracle = torch.index_select(out_oracle, dim=0, index=out_oracle_indices)
#   
#       # build density vector
#       num_gt_points = in_gt_depth.shape[0]
#       perc = np.array(list(range(5,105,5)))
#       density = np.array((perc/100) * (num_gt_points), dtype=np.int32)
#   
#       in_prec = np.zeros(density.shape)
#       in_prec_oracle = np.zeros(density.shape)
#       out_prec = np.zeros(density.shape)
#       out_prec_oracle = np.zeros(density.shape)
#   
#       in_rec = np.zeros(density.shape)
#       in_rec_oracle = np.zeros(density.shape)
#       out_rec = np.zeros(density.shape)
#       out_rec_oracle = np.zeros(density.shape)
#   
#       for i,k in enumerate(density):
#           # compute input absolute error chunk
#           iae = torch.abs(in_gt_depth[0:k] - in_depth[0:k])
#           num_inliers = torch.sum(torch.le(iae, th).to(torch.float32).to(device))
#   
#           in_prec[i] = num_inliers / k
#           in_rec[i] = num_inliers / valid_count
#   
#           # compute input oracle chunk
#           in_prec_oracle[i] = torch.sum(torch.le(in_oracle[0:k], th).to(torch.float32).to(device)) / k
#           in_rec_oracle[i] = torch.sum(torch.le(in_oracle[0:k], th).to(torch.float32).to(device)) / valid_count
#   
#           # compute output absolute error chunk
#           oae = torch.abs(out_gt_depth[0:k] - out_depth[0:k])
#           num_inliers = torch.sum(torch.le(oae, th).to(torch.float32).to(device))
#           out_prec[i] = num_inliers / k
#           out_rec[i] = num_inliers / valid_count
#   
#           # compute output oracle chunk
#           out_prec_oracle[i] = torch.sum(torch.le(out_oracle[0:k], th).to(torch.float32).to(device)) / k
#           out_rec_oracle[i] = torch.sum(torch.le(out_oracle[0:k], th).to(torch.float32).to(device)) / valid_count
#   
#       in_prec_str = ",".join([ "{:0.5f}".format(r) for r in in_prec ])
#       out_prec_str = ",".join([ "{:0.5f}".format(r) for r in out_prec ])
#       in_prec_oracle_str = ",".join([ "{:0.5f}".format(r) for r in in_prec_oracle ])
#       out_prec_oracle_str = ",".join([ "{:0.5f}".format(r) for r in out_prec_oracle ])
#   
#       in_rec_str = ",".join([ "{:0.5f}".format(r) for r in in_rec ])
#       out_rec_str = ",".join([ "{:0.5f}".format(r) for r in out_rec ])
#       in_rec_oracle_str = ",".join([ "{:0.5f}".format(r) for r in in_rec_oracle ])
#       out_rec_oracle_str = ",".join([ "{:0.5f}".format(r) for r in out_rec_oracle ])
#   
#       # store data
#       with open(stats_file,'a') as f:
#           f.write("{}\n".format(in_prec_str))
#           f.write("{}\n".format(out_prec_str))
#           f.write("{}\n".format(in_prec_oracle_str))
#           f.write("{}\n".format(out_prec_oracle_str))
#   
#           f.write("{}\n".format(in_rec_str))
#           f.write("{}\n".format(out_rec_str))
#           f.write("{}\n".format(in_rec_oracle_str))
#           f.write("{}\n".format(out_rec_oracle_str))
#   
#       return
