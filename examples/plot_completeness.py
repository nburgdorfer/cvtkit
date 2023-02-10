import numpy as np
import sys
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import argparse

# custom imports
from cvt.io import *
from cvt.camera import *
from cvt.util import *
from cvt.geometry import *
from cvt.metrics import *
from cvt.datasets.dtu import *

# argument parsing
parse = argparse.ArgumentParser(description="3D Chamfer Distance vs. Ground-Truth Visbility per pixel.")
parse.add_argument("-g", "--gt_depth_dir", default="./gt_depths", type=str, help="Path to ground-truth depth maps directory.")
parse.add_argument("-i", "--input_depth_dir", default="./input_depths", type=str, help="Path to input depth maps directory.")
parse.add_argument("-f", "--fused_depth_dir", default="./fused_depths", type=str, help="Path to fused depth maps directory.")
parse.add_argument("-c", "--camera_dir", default="./cameras", type=str, help="Path to camera files directory.")
parse.add_argument("-o", "--output_dir", default="./output", type=str, help="Path to desired output directory.")
parse.add_argument("-e", "--eval_data_dir", default="./eval_data", type=str, help="Path to the DTU evaluation data directory.")
parse.add_argument("-s", "--scan", default=1, type=int, help="The scan number being evaluated.")
parse.add_argument("-m", "--max_dist", default=5.0, type=int, help="Maximum error threshold for visualization.")
ARGS = parse.parse_args()

def main():
    # hard coded for now...
    s = 29
    network="gbinet"
    scan = "scan{:03d}".format(s)
    gt_depth_dir = "/media/nate/Data/Fusion/dtu/{}/GT_Depths/{}/".format(network, scan)
    input_depth_dir = "/media/nate/Drive1/Results/V-FUSE/dtu/Output_{}/{}/input_depths/".format(network, scan)
    fused_depth_dir = "/media/nate/Drive1/Results/V-FUSE/dtu/Output_{}/{}/depths/".format(network, scan)
    image_dir = "/media/nate/Data/Fusion/dtu/{}/Images/{}/".format(network, scan)
    camera_dir = "/media/nate/Drive1/Results/V-FUSE/dtu/Output_{}/{}/cams/".format(network, scan)
    output_dir = "/media/nate/Drive1/Results/V-FUSE/dtu/Output_{}/stats/{}/".format(network, scan)
    eval_data_dir = "/media/nate/Data/Evaluation/dtu/mvs_data/"
    max_dist = 0.15

    # extract arguments
    #   scan = "scan{:03d}".format(ARGS.scan)
    #   gt_depth_dir = ARGS.gt_depth_dir
    #   input_depth_dir = ARGS.input_depth_dir
    #   fused_depth_dir = ARGS.fused_depth_dir
    #   camera_dir = ARGS.camera_dir
    #   output_dir = ARGS.output_dir
    #   eval_data_dir = ARGS.eval_data_dir
    #   max_dist = ARGS.max_dist

    # build sorted lists of files
    gt_depth_files = os.listdir(gt_depth_dir)
    gt_depth_files.sort()
    gt_depth_files = [os.path.join(gt_depth_dir,g) for g in gt_depth_files if g[-9:]=="depth.pfm"]

    input_depth_files = os.listdir(input_depth_dir)
    input_depth_files.sort()
    input_depth_files = [os.path.join(input_depth_dir,i) for i in input_depth_files if i[-4:]==".pfm"]

    fused_depth_files = os.listdir(fused_depth_dir)
    fused_depth_files.sort()
    fused_depth_files = [os.path.join(fused_depth_dir,f) for f in fused_depth_files if f[-4:]==".pfm"]

    image_files = os.listdir(image_dir)
    image_files.sort()
    image_files = [os.path.join(image_dir,i) for i in image_files if i[-4:]==".png"]

    cam_files = os.listdir(camera_dir)
    cam_files.sort()
    cam_files = [os.path.join(camera_dir,c) for c in cam_files if c[-4:]==".txt"]

    total_views = len(fused_depth_files)

    # build view-colored point clouds
    input_ply = o3d.geometry.PointCloud()
    fused_ply = o3d.geometry.PointCloud()
    view_vecs = np.zeros((total_views, 3))
    with tqdm(fused_depth_files, unit="views") as data_loader:
        for view_num,fdf in enumerate(data_loader):
            # load data
            idf = input_depth_files[view_num]
            cf = cam_files[view_num]
            input_depth = read_pfm(idf)
            fused_depth = read_pfm(fdf)
            cam = read_single_cam_sfm(cf)

            # build per-view ply (points encoded with view number)
            view_vecs[view_num] = np.asarray([(view_num/total_views), 0, 1-(view_num/total_views)])
            input_view_ply = point_cloud_from_depth(input_depth, cam, view_vecs[view_num])
            input_ply += input_view_ply

            fused_view_ply = point_cloud_from_depth(fused_depth, cam, view_vecs[view_num])
            fused_ply += fused_view_ply
    
    print("building points filters...")
    # read in gt point cloud
    gt_ply_filename = "stl{:03d}_total.ply".format(s)
    gt_ply_path = os.path.join(eval_data_dir, "Points", "stl", gt_ply_filename)
    gt_ply = read_point_cloud(gt_ply_path)

    # build points filter based on input mask
    input_ply = filter_outlier_points(input_ply, gt_ply, 20)
    input_filt = build_est_points_filter(input_ply, eval_data_dir, s)

    fused_ply = filter_outlier_points(fused_ply, gt_ply, 20)
    fused_filt = build_est_points_filter(fused_ply, eval_data_dir, s)

    gt_filt = build_gt_points_filter(gt_ply, eval_data_dir, s)

    
    # compare point clouds
    print("Computing completeness...")
    input_comp_points, input_comp_dists, input_comp_colors = completeness_eval(input_ply, gt_ply, 20, input_filt, gt_filt)
    fused_comp_points, fused_comp_dists, fused_comp_colors = completeness_eval(fused_ply, gt_ply, 20, fused_filt, gt_filt)


    fused_depth = read_pfm(fused_depth_files[0])
    shape = fused_depth.shape
    with tqdm(cam_files, unit="views") as data_loader:
        for view_num,cf in enumerate(data_loader):
            # load data
            cam = read_single_cam_sfm(cf)
            image = cv2.imread(image_files[view_num])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image,2)
            image = np.repeat(image, 3, axis=2)

            # get projected input view completeness
            input_cloud = o3d.geometry.PointCloud()
            input_view_inds = np.where(np.all(input_comp_colors == view_vecs[view_num], axis=2))[0]
            input_cloud.points = o3d.utility.Vector3dVector(np.squeeze(input_comp_points[input_view_inds], axis=1))
            input_view_dists = input_comp_dists[input_view_inds]
            cmap = plt.get_cmap("cool")
            colors = cmap(np.minimum(input_view_dists, max_dist) / max_dist)[:,0,:3]
            input_cloud.colors = o3d.utility.Vector3dVector(colors)
            input_comp_map = render_point_cloud(input_cloud, cam, shape[1], shape[0])
            sum_map = np.sum(input_comp_map, axis=2).flatten()
            empty_pixs = np.argwhere(sum_map < 10.0)
            input_comp_map = input_comp_map.reshape(-1,3)
            image = image.reshape(-1,3)
            input_comp_map[empty_pixs] = image[empty_pixs]
            input_comp_map = input_comp_map.reshape(shape[0],shape[1],3)
            fname = os.path.join(output_dir,"{:04d}_comp_input.png".format(view_num))
            cv2.imwrite(fname, input_comp_map)



            # get projected fused view completeness
            fused_cloud = o3d.geometry.PointCloud()
            fused_view_inds = np.where(np.all(fused_comp_colors == view_vecs[view_num], axis=2))[0]
            fused_cloud.points = o3d.utility.Vector3dVector(np.squeeze(fused_comp_points[fused_view_inds], axis=1))
            fused_view_dists = fused_comp_dists[fused_view_inds]
            cmap = plt.get_cmap("cool")
            colors = cmap(np.minimum(fused_view_dists, max_dist) / max_dist)[:,0,:3]
            fused_cloud.colors = o3d.utility.Vector3dVector(colors)
            fused_comp_map = render_point_cloud(fused_cloud, cam, shape[1], shape[0])
            sum_map = np.sum(fused_comp_map, axis=2).flatten()
            empty_pixs = np.argwhere(sum_map < 10.0)
            fused_comp_map = fused_comp_map.reshape(-1,3)
            image = image.reshape(-1,3)
            fused_comp_map[empty_pixs] = image[empty_pixs]
            fused_comp_map = fused_comp_map.reshape(shape[0],shape[1],3)
            fname = os.path.join(output_dir,"{:04d}_comp_fused.png".format(view_num))
            cv2.imwrite(fname, fused_comp_map)

if __name__=="__main__":
    main()
