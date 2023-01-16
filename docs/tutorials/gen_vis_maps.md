## Generate Multi-View Visibility Maps
This script generates per-view visibility maps given a set of ground-truth depth maps with corresponding camera parameters. 

```python
import numpy as np
import sys
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import argparse

from cvt.common import *
from cvt.geometry import *
from cvt.metrics import *

# argument parsing
parse = argparse.ArgumentParser(description="Ground-Truth Visbility map generator.")
parse.add_argument("-g", "--gt_depth_dir", default="./gt_depths", type=str, help="Path to ground-truth depth maps directory.")
parse.add_argument("-c", "--camera_dir", default="./cameras", type=str, help="Path to camera files directory.")
parse.add_argument("-o", "--output_dir", default="./output", type=str, help="Path to desired output directory.")
parse.add_argument("-s", "--scan", default=1, type=int, help="The scan number being evaluated.")
ARGS = parse.parse_args()

def main():
    # extract arguments
    scan = "scan{:03d}".format(ARGS.scan)
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
    view_vecs = np.zeros((total_views, 3))
    with tqdm(gt_depth_files, unit="views") as data_loader:
        for view_num,gdf in enumerate(data_loader):
            # load data
            cf = cam_files[view_num]
            gt_depth = read_pfm(gdf)
            cam = read_cam(open(cf,'r'))

            # compute visibility scores
            gt_vis_map = visibility(gt_depth, cam, gt_depth_files, cam_files, view_num, pix_th=0.5)
            write_pfm(os.path.join(output_dir, "{:08d}.pfm".format(view_num)), gt_vis_map)
            cv2.imwrite(os.path.join(output_dir, "{:08d}.png".format(view_num)), (gt_vis_map/np.max(gt_vis_map)*255))

    
if __name__=="__main__":
    main()
```
