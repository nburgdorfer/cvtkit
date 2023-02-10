# cvt/datasets/dtu.py
"""Module including functions specific to the DTU dataset.

This module includes the following funcitons:

- ` build_est_points_filter(est_ply, data_path, scan_num)` - Builds a filter for an estimated point cloud.
- ` build_gt_points_filter(ply, data_path, scan_num)` - Constructs a points filter for a ground-truth point cloud.
"""
import os
import sys
import scipy.io as sio
import numpy as np
import open3d as o3d

from util import *

def build_est_points_filter(est_ply: o3d.geometry.PointCloud, data_path: str, scan_num: int) -> np.ndarray:
    """Builds a filter for an estimated point cloud.

    Parameters
        est_ply: Estimated point cloud for which a filter is being built.
        data_path: Data path to the DTU evaluation data.
        scan_num: DTU scan number for the point cloud.

    Returns:
        The points filter for the given point cloud.
    """
    # read in matlab bounding box, mask, and resolution
    mask_filename = "ObsMask{}_10.mat".format(scan_num)
    mask_path = os.path.join(data_path, "ObsMask/", mask_filename)
    data = sio.loadmat(mask_path)
    bounds = np.asarray(data["BB"])
    min_bound = bounds[0,:]
    max_bound = bounds[1,:]
    mask = np.asarray(data["ObsMask"])
    res = int(data["Res"])

    points = np.asarray(est_ply.points).transpose()
    shape = points.shape
    mask_shape = mask.shape
    filt = np.zeros(shape[1])

    min_bound = min_bound.reshape(3,1)
    min_bound = np.tile(min_bound, (1,shape[1]))

    qv = points
    qv = (points - min_bound) / res
    qv = np.rint(qv).astype(int)

    # get all valid points
    in_bounds = np.asarray(np.where( ((qv[0,:]>=0) & (qv[0,:] < mask_shape[0]) & (qv[1,:]>=0) & (qv[1,:] < mask_shape[1]) & (qv[2,:]>=0) & (qv[2,:] < mask_shape[2])))).squeeze(0)
    valid_points = qv[:,in_bounds]

    # convert 3D coords ([x,y,z]) to appropriate flattened coordinate ((x*mask_shape[1]*mask_shape[2]) + (y*mask_shape[2]) + z )
    mask_inds = np.ravel_multi_index(valid_points, dims=mask.shape, order='C')

    # further trim down valid points by mask value (keep point if mask is True)
    mask = mask.flatten()
    valid_mask_points = np.asarray(np.where(mask[mask_inds] == True)).squeeze(0)

    # add 1 to indices where we want to keep points
    filt[in_bounds[valid_mask_points]] = 1

    return filt

def build_gt_points_filter(ply: o3d.geometry.PointCloud, data_path: str, scan_num: int) -> np.ndarray:
    """Constructs a points filter for a ground-truth point cloud.

    Parameters:
        ply: Ground-truth point cloud for which a filter is being built.
        data_path: Data path to the DTU evaluation data.
        scan_num: DTU scan number for the point cloud.

    Returns:
        The points filter for the given point cloud.
    """
    # read in matlab gt plane 
    mask_filename = "Plane{}.mat".format(scan_num)
    mask_path = os.path.join(data_path, "ObsMask", mask_filename)
    data = sio.loadmat(mask_path)
    P = np.asarray(data["P"])

    points = np.asarray(ply.points).transpose()
    shape = points.shape

    # compute iner-product between points and the defined plane
    Pt = P.transpose()

    points = np.concatenate((points, np.ones((1,shape[1]))), axis=0)
    plane_prod = (Pt @ points).squeeze(0)

    # get all valid points
    filt = np.asarray(np.where((plane_prod > 0), 1, 0))

    return filt
