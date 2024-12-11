# cvt/metrics.py

"""Module including routines computing metrics.

This module includes the following functions:
"""

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from typing import Tuple, Optional
import torch

def MAE(estimate, target, reduction_dims, mask=None, relative=False):
    """Mean Absolute Error.

    Parameters:
        estimate: .
        target: .
        reduction_dims: .
        mask: .
        relative: .

    Returns: .
    """
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
    """Root Mean Squared Error.

    Parameters:
        estimate: .
        target: .
        mask: .
        relative: .

    Returns: .
    """
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
