import numpy as np
from sklearn.neighbors import KDTree


def accuracy_eval(est_ply, gt_ply, mask_th, max_dist, min_dist, est_filt=None, gt_filt=None):
    mask_gt = 20.0
    inlier_th = 0.5

    ### compute bi-directional chamfer distance between point clouds ###
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


def completeness_eval(est_ply, gt_ply, mask_th, max_dist, min_dist, est_filt=None, gt_filt=None):
    mask_gt = 20.0
    inlier_th = 0.5

    # build KD-Tree of estimated point cloud for querying
    tree = KDTree(np.asarray(est_ply.points), leaf_size=40)
    (dists, inds) = tree.query(np.asarray(gt_ply.points), k=1)


    # extract valid indices
    valid_inds = set(np.where(gt_filt == 1)[0])
    valid_dists = set(np.where(dists <= mask_gt)[0])
    valid_inds.intersection_update(valid_dists)
    valid_inds = np.asarray(list(valid_inds))

    dists = dists[valid_inds]
    inds = inds[valid_inds]
    ply_points = np.asarray(est_ply.points)[inds]
    colors = np.asarray(est_ply.colors)[inds]

    return ply_points, dists, colors
