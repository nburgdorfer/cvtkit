import numpy as np
import sys
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import argparse

from cvt.camera import *
from cvt.io import *
from cvt.geometry import *
from cvt.metrics import *

# argument parsing
parse = argparse.ArgumentParser(description="Ground-Truth Visbility map generator.")
parse.add_argument("-g", "--gt_depth_dir", default="./gt_depths", type=str, help="Path to ground-truth depth maps directory.")
parse.add_argument("-c", "--camera_dir", default="./cameras", type=str, help="Path to camera files directory.")
parse.add_argument("-o", "--output_dir", default="./output", type=str, help="Path to desired output directory.")
parse.add_argument("-s", "--scan", default=1, type=int, help="The scan number being evaluated.")
parse.add_argument("-v", "--voxel_size", default=0.03, type=float, help="Voxel size for downsampling point cloud.")
ARGS = parse.parse_args()

def main():
    # extract arguments
    scan = "scan{:03d}".format(ARGS.scan)
    cloud_name = "stl{:03d}_total.ply".format(ARGS.scan)
    gt_depth_dir = ARGS.gt_depth_dir
    camera_dir = ARGS.camera_dir
    output_dir = ARGS.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # build sorted lists of files
    gt_depth_files = os.listdir(gt_depth_dir)
    gt_depth_files.sort()
    gt_depth_files = [os.path.join(gt_depth_dir,g) for g in gt_depth_files if g[-9:]=="depth.pfm"]

    cam_files = os.listdir(camera_dir)
    cam_files.sort()
    cam_files = [os.path.join(camera_dir,c) for c in cam_files if c[-4:]==".txt"]

    total_views = len(gt_depth_files)

    # build view-colored point clouds
    cloud = o3d.geometry.PointCloud()
    with tqdm(gt_depth_files, unit="views") as data_loader:
        for view_num,gdf in enumerate(data_loader):
            # load data
            cf = cam_files[view_num]
            gt_depth = read_pfm(gdf)
            cam = read_single_cam_sfm(cf)

            # compute dense pc
            cloud += point_cloud_from_depth(gt_depth, cam, np.asarray([0,0,0]))
            
    cloud = cloud.voxel_down_sample(voxel_size=ARGS.voxel_size)
    o3d.io.write_point_cloud(os.path.join(ARGS.output_dir, cloud_name), cloud, write_ascii=False, compressed=True)


    
if __name__=="__main__":
    main()
