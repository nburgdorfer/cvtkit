import numpy as np
import sys
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import argparse
import torch

from cvtkit.camera import *
from cvtkit.io import read_cams_sfm, write_pfm, read_pfm
from cvtkit.geometry import visibility_numpy
from cvtkit.metrics import *

# argument parsing
parse = argparse.ArgumentParser(description="Ground-Truth Visbility map generator.")
parse.add_argument("-g", "--gt_depth_dir", default="./gt_depths", type=str, help="Path to ground-truth depth maps directory.")
parse.add_argument("-c", "--camera_dir", default="./cameras", type=str, help="Path to camera files directory.")
parse.add_argument("-o", "--output_dir", default="./output", type=str, help="Path to desired output directory.")
ARGS = parse.parse_args()

def main():
    
    scans = os.listdir(ARGS.gt_depth_dir)
    scans.sort()

    for scan in scans:
        gt_depth_dir = os.path.join(ARGS.gt_depth_dir,scan)
        camera_dir = ARGS.camera_dir
        output_dir = os.path.join(ARGS.output_dir, scan)

        os.makedirs(output_dir, exist_ok=True)

        # build sorted lists of files
        gt_depth_files = os.listdir(gt_depth_dir)
        gt_depth_files.sort()
        gt_depths = [read_pfm(os.path.join(gt_depth_dir,g)) for g in gt_depth_files if g[-9:]=="depth.pfm"]
        gt_depths = np.asarray(gt_depths)

        # load cameras
        cams = read_cams_sfm(camera_dir)

        total_views = gt_depths.shape[0]
        with tqdm(range(total_views), unit="views") as loader:
            for view_num in loader:
                gt_vis_map = visibility_numpy(gt_depths, reference_index=view_num, K=cams[view_num,1,:3,:3], Ps=cams[:,0], pix_th=1.0)
                write_pfm(os.path.join(output_dir, "{:08d}.pfm".format(view_num)), gt_vis_map)
                cv2.imwrite(os.path.join(output_dir, "{:08d}.png".format(view_num)), (gt_vis_map/(total_views)*255))

if __name__=="__main__":
    main()
