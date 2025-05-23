# cvt/visualization/util.py
"""Module including general utilities for visualization.

This module includes the following functions:


"""
import open3d as o3d
import numpy as np
import cv2
import scipy.ndimage as ndimage
import skimage.transform as transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import torch
import torch.nn.functional as F
import os,sys
import torchvision.transforms.functional as tvf
import torchvision.transforms as tvt

from common import non_zero_std
from io import *

def display_inlier_outlier(cloud: o3d.geometry.PointCloud, indices: np.ndarray) -> None:
    """Displays a point cloud with outlier points colored red.

    Parameters:
        cloud: Point cloud to be displayed.
        indices: Indices indicating the inlier points.
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

    return

def print_csv(data):
    for i,d in enumerate(data):
        if i==len(data)-1:
            print(f"{d:6.4f}")
        else:
            print(f"{d:6.4f}", end=",")

def plot_cameras(cams, num_cams, scale, A, output_file):
    # grab the requested number of cameras and apply the alignment
    P = cams[:num_cams]
    P = np.array([ A @ np.linalg.inv(p) for p in P ])

    # create 'num_cams' intrinsic matrices (just used for point-cloud camera pyramid geometry)
    k = np.array([[233.202, 0.0, 144.753],[0.0, 233.202, 108.323],[0.0, 0.0, 1.0]])
    K = np.array([ k for p in P])

    # build list of camera pyramid points
    pyr_pts = []
    for k,p in zip(K,P):
        pyr_pt = build_cam_pyr(scale, k)
        pyr_pt = p @ pyr_pt
        pyr_pts.append(pyr_pt)

    # build point cloud using camera centers
    build_pyr_point_cloud(pyr_pts, output_file)

    return


f_matrix = None
img2_line = None
img1 = None
img2 = None

CENTERED = False


def mouse1_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        #print("mouse1 X: " + str(x) + " Y: " + str(y))
#        y = img1.shape[0] - y
        if CENTERED:
            x -= img1.shape[1] / 2
            y -= img1.shape[0] / 2
        mouse1_pt = np.asarray([x, y, 1.0])
        i2ray = np.dot(f_matrix, mouse1_pt)
        if CENTERED:
            i2ray[2] = i2ray[2] - i2ray[0]*img2.shape[1]/2 - i2ray[1]*img2.shape[0]/2 
        #print(i2ray)
        i2pt1_x = 0
        i2pt2_x = img2.shape[1]
        i2pt1_y = int(-(i2ray[2] + i2ray[0] * i2pt1_x) / i2ray[1])
        i2pt2_y = int(-(i2ray[2] + i2ray[0] * i2pt2_x) / i2ray[1])

        global img2_line
#        img2_line = ((i2pt1_x, img1.shape[0] - i2pt1_y), (i2pt2_x, img1.shape[0] - i2pt2_y))
        img2_line = ((i2pt1_x, i2pt1_y), (i2pt2_x, i2pt2_y))

def draw_line(img, line):
    if line is None:
        return img
    # Clip the line to image bounds
    ret, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), line[0], line[1])
    if ret:
        img = img.copy()
        cv2.line(img, p1, p2, (255, 255, 0), 1)
    return img


def scale_img(img, scale):
    img_scaled = transform.resize(img, [img.shape[0]*scale, img.shape[1]*scale], mode="constant")
    return img_scaled.astype(np.float32)

def scale_f_mat(mat, scale):
    mat[:, 2] *= scale
    mat[2, :] *= scale
    return mat

def fmat_demo(img1l, img2l, fl, scale=1.0):
    global img1, img2, f_matrix

    img1 = scale_img(img1l, scale)
    img2 = scale_img(img2l, scale)
    f_matrix = scale_f_mat(fl, scale)

    f_matrix = f_matrix/np.max(f_matrix)

    print("Img1: " + str(img1.shape))
    print("Img2: " + str(img2.shape))

    cv2.namedWindow("img1")
    cv2.namedWindow("img2")
    cv2.setMouseCallback("img1", mouse1_callback)

    while True:
        show2 = draw_line(img2, img2_line)
        show1 = img1
        cv2.imshow("img1", cv2.cvtColor(show1, cv2.COLOR_BGR2RGB))
        cv2.imshow("img2", cv2.cvtColor(show2, cv2.COLOR_BGR2RGB))
        cv2.waitKey(50)

def build_pyr_point_cloud(pyr_pts, filename):
    """Builds a point cloud for a camera frustum visual.

    !!! bug "needs work..."
    """
    num_pts = len(pyr_pts)
    element_vertex = 6*num_pts
    element_edge = 10*num_pts

    with open(filename, 'w') as fh:
        # write header meta-data
        fh.write('ply\n')
        fh.write('format ascii 1.0\n')
        fh.write('comment Right-Handed System\n')
        fh.write('element vertex {}\n'.format(element_vertex))
        fh.write('property float x\n')
        fh.write('property float y\n')
        fh.write('property float z\n')
        fh.write('property uchar red\n')
        fh.write('property uchar green\n')
        fh.write('property uchar blue\n')
        fh.write('element edge {}\n'.format(element_edge))
        fh.write('property int vertex1\n')
        fh.write('property int vertex2\n')
        fh.write('property uchar red\n')
        fh.write('property uchar green\n')
        fh.write('property uchar blue\n')
        fh.write('end_header\n')

        # write vertex data to file
        for pt in pyr_pts:
            fh.write('{:.10f}'.format( pt[0][0][0] ) + ' ' + '{:.10f}'.format( pt[0][1][0] ) + ' ' + '{:.10f}'.format( pt[0][2][0] ) + ' 255 128 0\n')
            fh.write('{:.10f}'.format( pt[1][0][0] ) + ' ' + '{:.10f}'.format( pt[1][1][0] ) + ' ' + '{:.10f}'.format( pt[1][2][0] ) + ' 255 128 0\n')
            fh.write('{:.10f}'.format( pt[2][0][0] ) + ' ' + '{:.10f}'.format( pt[2][1][0] ) + ' ' + '{:.10f}'.format( pt[2][2][0] ) + ' 255 128 0\n')
            fh.write('{:.10f}'.format( pt[3][0][0] ) + ' ' + '{:.10f}'.format( pt[3][1][0] ) + ' ' + '{:.10f}'.format( pt[3][2][0] ) + ' 255 128 0\n')
            fh.write('{:.10f}'.format( pt[4][0][0] ) + ' ' + '{:.10f}'.format( pt[4][1][0] ) + ' ' + '{:.10f}'.format( pt[4][2][0] ) + ' 255 128 0\n')
            fh.write('{:.10f}'.format( pt[5][0][0] ) + ' ' + '{:.10f}'.format( pt[5][1][0] ) + ' ' + '{:.10f}'.format( pt[5][2][0] ) + ' 255 128 0\n')

        # write edge data to file
        for i in range(num_pts):
            edge_ind = i*6
            fh.write('{} {} 255 0 0\n'.format(edge_ind, edge_ind+1))
            fh.write('{} {} 255 0 0\n'.format(edge_ind, edge_ind+2))
            fh.write('{} {} 255 0 0\n'.format(edge_ind, edge_ind+3))
            fh.write('{} {} 255 0 0\n'.format(edge_ind, edge_ind+4))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+1, edge_ind+2))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+2, edge_ind+3))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+3, edge_ind+4))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+4, edge_ind+1))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+1, edge_ind+5))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+5, edge_ind+2))
    return

def build_cam_pyr(cam_scale, K):
    """Constructs a camera frustum for visualization.

    !!! bug "needs work..."
    """
    focallen   = K[0][0]
    cam_w      = 2 * K[0][2]
    cam_h      = 2 * K[1][2]
    cam_center = np.array([0.0,          0.0,          0.0,      1.0])
    cam_ul     = np.array([cam_w * -0.5, cam_h * -0.5, focallen, 1.0])
    cam_ur     = np.array([cam_w *  0.5, cam_h * -0.5, focallen, 1.0])
    cam_dr     = np.array([cam_w *  0.5, cam_h *  0.5, focallen, 1.0])
    cam_dl     = np.array([cam_w * -0.5, cam_h *  0.5, focallen, 1.0])
    cam_top    = np.array([0.0,          cam_h * -0.7, focallen, 1.0])
    cam_center *= cam_scale
    cam_ul     *= cam_scale
    cam_ur     *= cam_scale
    cam_dr     *= cam_scale
    cam_dl     *= cam_scale
    cam_top    *= cam_scale
    cam_center[3] = 1.0
    cam_ul[3]     = 1.0
    cam_ur[3]     = 1.0
    cam_dr[3]     = 1.0
    cam_dl[3]     = 1.0
    cam_top[3]    = 1.0
    cam_center = cam_center.reshape((4, 1))
    cam_ul     = cam_ul.reshape((4, 1))
    cam_ur     = cam_ur.reshape((4, 1))
    cam_dr     = cam_dr.reshape((4, 1))
    cam_dl     = cam_dl.reshape((4, 1))
    cam_top    = cam_top.reshape((4, 1))
    return [cam_center, cam_ul, cam_ur, cam_dr, cam_dl, cam_top]


def display_map(filename: str, disp_map: np.ndarray, mx: float, mn: float) -> None:
    """Writes an input map to a normalized image.

    Parameters:
        filename: Name of the file to store the input map.
        disp_map: Map to be stored as an image file.
        mx: maximum value used for pixel intensity normalization.
        mn: minimum value used for pixel intensity normalization.
    """
    disp_map = ((disp_map-mn)/(mx-mn+1e-8))*255
    cv2.imwrite(filename, disp_map)

def plot_coverage(data, output, batch_ind, vis_path):
    hypos = output["hypos"]
    intervals = output["intervals"]
    _,H,W = data["target_depth"].shape
    levels = len(hypos)

    uncovered_mask
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
        cov_percent = coverage.sum() / (valid_targets.sum() + 1e-10)
        coverage = coverage.reshape(h,w,1).repeat(1,1,3) * (torch.tensor([[[0,255,0]]]).to(coverage).repeat(h,w,1))
        uncovered = uncovered.reshape(h,w,1).repeat(1,1,3) * (torch.tensor([[[255,0,0]]]).to(uncovered).repeat(h,w,1))

        # plot
        total_coverage = torch.movedim(coverage+uncovered, (0,1,2), (1,2,0))
        total_coverage = torch.movedim(tvf.resize(total_coverage, [H,W], interpolation=tvt.InterpolationMode.NEAREST), (0,1,2), (2,0,1))
        plt.imshow((total_coverage).detach().cpu().numpy())
        plt.title(f"Coverage ({cov_percent*100:0.2f}%)")
        plt.axis('off')
        plt.savefig(os.path.join(vis_path, f"coverage_{batch_ind:08d}_l{l}.png"))
        plt.close()

def visualize_relative_variance(target_cost, cost_volume, hypos, intervals, target_depth, mask, batch_ind, vis_path, level):
    batch_size, _, planes, h, w = cost_volume.shape
    batch_ind = int(batch_ind.item())
    cv = cost_volume.squeeze(dim=1)
    mask = mask.reshape(1, h, w).repeat(batch_size, 1, 1)
    hypos = hypos.reshape(batch_size, planes, h, w)
    intervals = intervals.reshape(batch_size, planes, h, w)
    target_depth = target_depth.reshape(batch_size, 1, h, w)

    ### compute coverage
    diff = torch.abs(hypos - target_depth)
    min_interval = intervals[:,0:1] * 0.5 # intervals are bin widths, divide by 2 for radius
    coverage = torch.clip(torch.where(diff <= min_interval, 1, 0).sum(dim=1, keepdim=True), 0, 1)
    uncovered = torch.clip(torch.where(coverage <= 0, 1, 0).sum(dim=1, keepdim=True), 0, 1)
    valid_targets = torch.where(target_depth > 0, 1, 0)
    uncovered *= valid_targets
    cov_percent = coverage.sum() / (valid_targets.sum() + 1e-10)
    coverage = coverage.reshape(h,w,1).repeat(1,1,3) * (torch.tensor([[[0,255,0]]]).to(coverage).repeat(h,w,1))
    uncovered = uncovered.reshape(h,w,1).repeat(1,1,3) * (torch.tensor([[[255,0,0]]]).to(uncovered).repeat(h,w,1))
    plt.imshow((coverage+uncovered).detach().cpu().numpy())
    plt.title(f"Coverage ({cov_percent*100:0.2f}%)")
    plt.axis('off')
    plt.savefig(os.path.join(vis_path,f"coverage_{batch_ind:08d}_l{level}.png"))
    plt.close()

    min_interval = intervals[:,0:1].repeat(1, planes, 1, 1)
    tcv = target_cost.reshape(batch_size, 1, h, w).repeat(1, planes, 1, 1)
    mask = mask.reshape(batch_size, 1, h, w).repeat(1, planes, 1, 1)
    gt_bin_mask = torch.where(diff <= min_interval, 0, 1)
    mask *= gt_bin_mask
    inds = torch.stack(torch.where(mask > 0), dim=-1)
    var_diff = torch.div(tcv,cv+1e-10)[inds[:,0],inds[:,1],inds[:,2],inds[:,3]]

    # compute mean and standard-deviation
    sigma = torch.std(var_diff)
    mean = torch.mean(var_diff)
    # select only values in 3-sigma radius
    var_diff = var_diff[torch.where(var_diff >= (mean-(3*sigma)))[0]]
    var_diff = var_diff[torch.where(var_diff <= (mean+(3*sigma)))[0]]
    # plot
    flat_var = var_diff.detach().cpu().numpy().flatten()
    plt.hist(flat_var, bins=100)
    plt.axvline(flat_var.mean(), c="red")
    plt.savefig(os.path.join(vis_path,f"relative_{batch_ind:08d}_l{level}.png"))
    plt.close()

    return mean, cov_percent

def visualize_point_variance(target_cost, cost_volume, hypos, intervals, target_depth, mask, batch_ind, vis_path, level):
    batch_size, groups, planes, h, w = cost_volume.shape
    batch_ind = int(batch_ind.item())
    tcv = target_cost.mean(dim=1, keepdim=True)
    cv = cost_volume.mean(dim=1, keepdim=True)
    mask = mask.reshape(1, h, w).repeat(batch_size, 1, 1)
    hypos = hypos.reshape(batch_size, planes, h, w)
    intervals = intervals.reshape(batch_size, planes, h, w)
    target_depth = target_depth.reshape(batch_size, 1, h, w)

    ### compute coverage
    diff = torch.abs(hypos - target_depth)
    min_interval = intervals[:,0:1]
    coverage = torch.clip(torch.where(diff <= min_interval, 1, 0).sum(dim=1, keepdim=True), 0, 1)
    uncovered = torch.clip(torch.where(coverage <= 0, 1, 0).sum(dim=1, keepdim=True), 0, 1)
    valid_targets = torch.where(target_depth > 0, 1, 0)
    uncovered *= valid_targets
    cov_percent = coverage.sum() / (valid_targets.sum() + 1e-10)
    coverage = coverage.reshape(h,w,1).repeat(1,1,3) * (torch.tensor([[[0,255,0]]]).to(coverage).repeat(h,w,1))
    uncovered = uncovered.reshape(h,w,1).repeat(1,1,3) * (torch.tensor([[[255,0,0]]]).to(uncovered).repeat(h,w,1))
    plt.imshow((coverage+uncovered).detach().cpu().numpy())
    plt.title(f"Coverage ({cov_percent*100:0.2f}%)")
    plt.axis('off')
    plt.savefig(os.path.join(vis_path,f"coverage_{batch_ind:08d}_l{level}.png"))
    plt.close()

    ### compute correct matching variance histogram
    var_match = tcv.reshape(batch_size, h, w)
    inds = torch.stack(torch.where(mask > 0), dim=-1)
    var_match = var_match[inds[:,0], inds[:,1], inds[:,2]]
    # compute mean and standard-deviation
    match_sigma = torch.std(var_match)
    match_mean = torch.mean(var_match)
    # select only values in 3-sigma radius
    var_match = var_match[torch.where(var_match >= (match_mean-(3*match_sigma)))[0]]
    var_match = var_match[torch.where(var_match <= (match_mean+(3*match_sigma)))[0]]
    # plot
    flat_var = var_match.detach().cpu().numpy().flatten()
    plt.hist(flat_var, bins=100)
    plt.axvline(flat_var.mean(), c="red")
    plt.savefig(os.path.join(vis_path,f"match_{batch_ind:08d}_l{level}.png"))
    plt.close()

    ### compute incorrect matching variance histogram
    mask = mask.reshape(batch_size, 1, h, w).repeat(1, planes, 1, 1)
    inds = torch.stack(torch.where(mask > 0), dim=-1)
    diff = diff[inds[:,0], inds[:,1], inds[:,2], inds[:,3]]
    min_interval = min_interval[inds[:,0], 0, inds[:,2], inds[:,3]]
    cv = cv[inds[:,0], 0, inds[:,1], inds[:,2], inds[:,3]]
    inds = torch.stack(torch.where(diff > min_interval), dim=-1)
    var_mismatch = cv[inds]
    # compute mean and standard-deviation
    mismatch_sigma = torch.std(var_mismatch)
    mismatch_mean = torch.mean(var_mismatch)
    # select only values in 3-sigma radius
    var_mismatch = var_mismatch[torch.where(var_mismatch >= (mismatch_mean-(3*mismatch_sigma)))[0]]
    var_mismatch = var_mismatch[torch.where(var_mismatch <= (mismatch_mean+(3*mismatch_sigma)))[0]]
    # plot
    flat_var = var_mismatch.detach().cpu().numpy().flatten()
    plt.hist(flat_var, bins=100)
    plt.axvline(flat_var.mean(), c="red")
    plt.savefig(os.path.join(vis_path,f"mismatch_{batch_ind:08d}_l{level}.png"))
    plt.close()

    return match_mean, mismatch_mean

def visualize_ray_points(rays, ind, edge_color="255 0 0"):
    out_file = f"cam_vis/ray_points_{ind:04d}.ply"
    num_rays, num_points, _ = rays.shape

    with open(out_file, "w") as of:
        of.write("ply\n")
        of.write("format ascii 1.0\n")
        of.write("comment VCGLIB generated\n")
        of.write(f"element vertex {num_rays*num_points}\n")
        of.write("property float x\n")
        of.write("property float y\n")
        of.write("property float z\n")
        of.write("property uchar red\n")
        of.write("property uchar green\n")
        of.write("property uchar blue\n")
        of.write("end_header\n")
        for i in range(num_rays):
            for j in range(num_points):
                of.write(f"{rays[i,j,0]:0.3f} {rays[i,j,1]:0.3f} {rays[i,j,2]:0.3f} 0 0 0\n")

def visualize_camera_frustum(planes, ind, edge_color="255 0 0"):
    out_file = f"cam_vis/cam_frustum_{ind:04d}.ply"
    num_planes, _, _ = planes.shape
    num_verts = 4*num_planes
    num_edges = 8*num_planes - 4

    with open(out_file, "w") as of:
        of.write("ply\n")
        of.write("format ascii 1.0\n")
        of.write("comment VCGLIB generated\n")
        of.write(f"element vertex {num_verts}\n")
        of.write("property float x\n")
        of.write("property float y\n")
        of.write("property float z\n")
        of.write("property uchar red\n")
        of.write("property uchar green\n")
        of.write("property uchar blue\n")
        of.write(f"element edge {num_edges}\n")
        of.write("property int vertex1\n")
        of.write("property int vertex2\n")
        of.write("property uchar red\n")
        of.write("property uchar green\n")
        of.write("property uchar blue\n")
        of.write("end_header\n")
        for p in range(num_planes):
            for i in range(4):
                of.write(f"{planes[p,0,i]:0.3f} {planes[p,1,i]:0.3f} {planes[p,2,i]:0.3f} 0 0 0\n")

        # draw plane border
        for p in range(num_planes):
            ind = p*4
            of.write(f"{ind} {ind+1} {edge_color}\n")
            of.write(f"{ind} {ind+2} {edge_color}\n")
            of.write(f"{ind+1} {ind+3} {edge_color}\n")
            of.write(f"{ind+2} {ind+3} {edge_color}\n")

        # draw plane connections
        for p in range(num_planes-1):
            ind = p*4
            of.write(f"{ind} {ind+4} {edge_color}\n")
            of.write(f"{ind+1} {ind+5} {edge_color}\n")
            of.write(f"{ind+2} {ind+6} {edge_color}\n")
            of.write(f"{ind+3} {ind+7} {edge_color}\n")

def visualize_bounding_box(min_bounds, max_bounds, output_file, edge_color="255 0 0"):

    num_verts = 8
    num_edges = 12
    with open(out_file, "w") as of:
        of.write("ply\n")
        of.write("format ascii 1.0\n")
        of.write("comment VCGLIB generated\n")
        of.write(f"element vertex {num_verts}\n")
        of.write("property float x\n")
        of.write("property float y\n")
        of.write("property float z\n")
        of.write("property uchar red\n")
        of.write("property uchar green\n")
        of.write("property uchar blue\n")
        of.write(f"element edge {num_edges}\n")
        of.write("property int vertex1\n")
        of.write("property int vertex2\n")
        of.write("property uchar red\n")
        of.write("property uchar green\n")
        of.write("property uchar blue\n")
        of.write("end_header\n")
        for p in range(num_planes):
            for i in range(4):
                of.write(f"{planes[p,0,i]:0.3f} {planes[p,1,i]:0.3f} {planes[p,2,i]:0.3f} 0 0 0\n")

        # draw plane border
        for p in range(num_planes):
            ind = p*4
            of.write(f"{ind} {ind+1} {edge_color}\n")
            of.write(f"{ind} {ind+2} {edge_color}\n")
            of.write(f"{ind+1} {ind+3} {edge_color}\n")
            of.write(f"{ind+2} {ind+3} {edge_color}\n")

        # draw plane connections
        for p in range(num_planes-1):
            ind = p*4
            of.write(f"{ind} {ind+4} {edge_color}\n")
            of.write(f"{ind+1} {ind+5} {edge_color}\n")
            of.write(f"{ind+2} {ind+6} {edge_color}\n")
            of.write(f"{ind+3} {ind+7} {edge_color}\n")



def visualize_mvs(data, output, batch_ind, vis_path, max_depth_error, mode, epoch=-1):
    image = torch.movedim(data["images"][0,0], (0,1,2), (2,0,1)).detach().cpu().numpy()
    image = ((image-image.min()) / (image.max()-image.min()+1e-10))

    target_depth = data["target_depth"].detach().cpu().numpy()[0,0]
    est_depth = output["final_depth"].detach().cpu().numpy()[0,0]
    est_conf = output["confidence"].detach().cpu().numpy()[0,0]

    assert(est_depth.shape == target_depth.shape)

    num_valid_pixels = np.where(target_depth > 0, 1, 0).sum()

    ## compute depth residual
    depth_residual = np.abs(est_depth - target_depth)
    res_temp = np.clip(depth_residual / (max_depth_error*3+1e-10), 0, 1)
    depth_residual[target_depth == 0.0] = 0.0
    depth_mae = np.mean(depth_residual)

    ## compute ROC and AUC
    perc, oracle_roc, est_roc, rel_auc = auc_score(data, output)

    ### plot coverage
    #levels = uncovered_masks.shape[0]
    #fig, axs = plt.subplots(1, levels)
    #fig.tight_layout()
    #for i in range(levels):
    #    coverage_percent = 1 - (uncovered_masks[i].sum() / num_valid_pixels)
    #    axs[i].imshow(uncovered_masks[i], cmap="gray", vmin=0.0, vmax=1.0)
    #    axs[i].set_title(f"Level {i}: \n{coverage_percent*100:0.2f}%")
    #    axs[i].set_xticks([])
    #    axs[i].set_yticks([])
    #plt.subplots_adjust(wspace=0, hspace=0.2)
    #plot_file = os.path.join(vis_path, f"{batch_ind:08d}_cov.png")
    #plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.4, dpi=400)
    #plt.clf()
    #plt.close()

    ## plot
    fig, axs = plt.subplots(2, 3)
    fig.tight_layout()
    max_depth = np.max(target_depth)
    # Row #1: Depth
    axs[0, 0].imshow(target_depth, cmap="gray", vmin=0, vmax=max_depth)
    axs[0, 0].set_title('Target Depth')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 1].imshow(est_depth, cmap="gray", vmin=0, vmax=max_depth)
    axs[0, 1].set_title('Est. Depth')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 2].imshow(depth_residual, cmap="hot", vmin=0, vmax=max_depth_error)
    axs[0, 2].set_title(f'Residual (mae: {depth_mae:0.3f}mm)')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    # Row #2: Confidence
    axs[1, 0].imshow(image)
    axs[1, 0].set_title('Image')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].imshow(est_conf, cmap="gray", vmin=0, vmax=1.0)
    axs[1, 1].set_title('Est. Confidence')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 2].plot(perc, oracle_roc, label="Oracle")
    axs[1, 2].plot(perc, est_roc, label="Confidence")
    axs[1, 2].set_title(f'ROC (rel. auc: {rel_auc:0.2f})')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].legend()
    plt.subplots_adjust(wspace=0, hspace=0.2)

    if epoch!=None and epoch >= 0:
        plot_file = os.path.join(vis_path, f"{mode}_{epoch:03d}_{batch_ind:08d}.png")
    else:
        plot_file = os.path.join(vis_path, f"{batch_ind:08d}.png")
    plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.4, dpi=400)
    plt.clf()
    plt.close()

def auc_score(data, output):
    est_depth = output["final_depth"][0,0]
    est_conf = output["confidence"][0,0]
    target_depth = data["target_depth"][0,0]

    mask = torch.where(target_depth > 0, 1.0, 0.0)
    pixel_count = int(torch.sum(mask).item())

    inds = torch.where(mask==1)
    target_depth = target_depth[inds].detach().cpu().numpy()
    est_depth = est_depth[inds].detach().cpu().numpy()
    est_conf = est_conf[inds].detach().cpu().numpy()

    # flatten to 1D tensor
    target_depth = target_depth.flatten()
    est_depth = est_depth.flatten()
    est_conf = est_conf.flatten()

    # compute error
    error = np.abs(est_depth - target_depth)
    
    # sort orcale curves by error
    oracle_indices = np.argsort(error)
    oracle_error = np.take(error, indices=oracle_indices, axis=0)

    # sort all tensors by confidence value
    est_indices_conf = np.argsort(est_conf)
    est_indices_conf = est_indices_conf[::-1]
    est_error_conf = np.take(error, indices=est_indices_conf, axis=0)

    # build density vector
    perc = np.array(list(range(5,105,5)))
    density = np.array((perc/100) * (pixel_count), dtype=np.int32)

    oracle_roc = np.zeros(density.shape)
    est_roc = np.zeros(density.shape)
    for i,k in enumerate(density):
        oe = oracle_error[:k]
        ee = est_error_conf[:k]

        if (oe.shape[0] == 0):
            oracle_roc[i] = 0.0
            est_roc[i] = 0.0
        else:
            oracle_roc[i] = np.mean(oe)
            est_roc[i] = np.mean(ee)

    # comput AUC
    oracle_auc = np.trapz(oracle_roc, dx=1)
    est_auc = np.trapz(est_roc, dx=1)

    return perc, oracle_roc, est_roc, (est_auc/oracle_auc)


def laplacian_count(data, output, plot_file=None, use_est_depth=False):
    target_depth = data["target_depth"].detach().cpu().numpy()[0]
    image_laplacian = output["image_laplacian"][0].detach().cpu().numpy()
    if use_est_depth:
        depth_laplacian = output["est_depth_laplacian"][0].detach().cpu().numpy()
    else:
        depth_laplacian = data["depth_laplacian"][0].detach().cpu().numpy()

    # Image Laplacian vs. Depth Laplacian Count
    inds = np.argwhere(target_depth.flatten() > 0.0)
    num_pix = int(inds.shape[0])
    il = (image_laplacian.flatten())[inds][:,0]
    dl = (depth_laplacian.flatten())[inds][:,0]
    M = np.zeros((5,5))
    for i in range(5):
        for d in range(5):
            M[i,d] = (int(((il == i) & (dl == d)).sum()) / num_pix)

    if plot_file != None:
        fig,ax = plt.subplots()
        img = ax.imshow(M, interpolation="none", cmap="copper")
        ax.set_xlabel("Est. Depth Laplacian")
        ax.set_ylabel("Image Laplacian")
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        for i in range(5):
            for j in range(5):
                text = ax.text(i, j, f"{(M[j, i]*100):0.2f}%", ha="center", va="center", color="w")
        plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.4, dpi=200)
        plt.clf()
        plt.close()

    return M


def laplacian_uncovered_count(data, output, plot_file=None):
    image_laplacian = output["image_laplacian"][0].detach().cpu().numpy()
    depth_laplacian = data["depth_laplacian"][0].detach().cpu().numpy()
    uncovered_masks = output["uncovered_masks"]

    # Image Laplacian vs. Depth Laplacian vs. Uncovered Count
    for mask in uncovered_masks:
        inds = np.argwhere(mask.flatten() > 0.0)
        num_pix = int(inds.shape[0])
        il = (image_laplacian.flatten())[inds][:,0]
        dl = (depth_laplacian.flatten())[inds][:,0]
        M = np.zeros((5,5))
        for i in range(5):
            for d in range(5):
                M[i,d] += (int(((il == i) & (dl == d)).sum()) / num_pix)

    if plot_file != None:
        fig,ax = plt.subplots()
        img = ax.imshow(M, interpolation="none", cmap="copper")
        ax.set_xlabel("Depth Laplacian")
        ax.set_ylabel("Image Laplacian")
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        for i in range(5):
            for j in range(5):
                text = ax.text(i, j, f"{M[j, i]:0.2f}", ha="center", va="center", color="w")
        plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.4, dpi=200)
        plt.clf()
        plt.close()

    return M

def laplacian_depth_error(data, output, plot_file=None, use_est_depth=False):
    target_depth = data["target_depth"].detach().cpu().numpy()[0,0]
    est_depth = output["final_depth"].detach().cpu().numpy()[0,0]
    image_laplacian = output["image_laplacian"][0].detach().cpu().numpy()
    if use_est_depth:
        depth_laplacian = output["est_depth_laplacian"][0].detach().cpu().numpy()
    else:
        depth_laplacian = data["depth_laplacian"][0].detach().cpu().numpy()

    abs_err = np.abs((est_depth - target_depth))
    inds = np.argwhere(target_depth.flatten() > 0.0)
    err = (abs_err.flatten())[inds]
    il = (image_laplacian.flatten())[inds][:,0]
    dl = (depth_laplacian.flatten())[inds][:,0]

    M = np.zeros((5,5))
    for i in range(5):
        for d in range(5):
            M[i,d] = err[np.argwhere((il == i) & (dl == d))].mean()

    if plot_file != None:
        fig,ax = plt.subplots()
        img = ax.imshow(M, interpolation="none", cmap="copper")
        ax.set_xlabel("Depth Laplacian")
        ax.set_ylabel("Image Laplacian")
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        for i in range(5):
            for j in range(5):
                text = ax.text(i, j, f"{M[j, i]:0.2f}", ha="center", va="center", color="w")
        plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.4, dpi=200)
        plt.clf()
        plt.close()

    return M

def plot_laplacian_matrix(M, plot_file, use_est_depth=False, count=False):
    fig,ax = plt.subplots()
    img = ax.imshow(M, interpolation="none", cmap="copper")
    if use_est_depth:
     ax.set_xlabel("Est. Depth Laplacian")
    else:
     ax.set_xlabel("Depth Laplacian")
    ax.set_ylabel("Image Laplacian")
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    for i in range(5):
     for j in range(5):
         if count:
             text = ax.text(i, j, f"{(M[j, i]*100):0.2f}%", ha="center", va="center", color="w")
         else:
             text = ax.text(i, j, f"{M[j, i]:0.2f}", ha="center", va="center", color="w")
    plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.4, dpi=200)
    plt.clf()
    plt.close()


def to_normal(ply_file, output_file, radius=15.0, max_nn=100):
    cloud = o3d.io.read_point_cloud(ply_file)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normals = np.asarray(cloud.normals)
    #normals = (normals-normals.min(axis=0)) / (normals.max(axis=0)-normals.min(axis=0))
    normals = (normals-normals.min()) / (normals.max()-normals.min())
    cloud.colors = o3d.utility.Vector3dVector(normals[:,::-1])
    o3d.io.write_point_cloud(output_file, cloud)
