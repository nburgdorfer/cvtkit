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
FILE_ROOT=os.path.abspath(os.path.dirname(__file__))
PYTHON_ROOT=os.path.abspath(os.path.dirname(FILE_ROOT))
SRC_ROOT=os.path.abspath(os.path.dirname(PYTHON_ROOT))

sys.path.append(PYTHON_ROOT)

from common import *
from geometry import *
from metrics import *

# argument parsing
parse = argparse.ArgumentParser(description="3D Chamfer Distance vs. Ground-Truth Visbility per pixel.")
parse.add_argument("-g", "--gt_depth_dir", default="./gt_depths", type=str, help="Path to ground-truth depth maps directory.")
parse.add_argument("-i", "--input_depth_dir", default="./input_depths", type=str, help="Path to input depth maps directory.")
parse.add_argument("-f", "--fused_depth_dir", default="./fused_depths", type=str, help="Path to fused depth maps directory.")
parse.add_argument("-c", "--camera_dir", default="./cameras", type=str, help="Path to camera files directory.")
parse.add_argument("-o", "--output_dir", default="./output", type=str, help="Path to desired output directory.")
parse.add_argument("-e", "--eval_data_dir", default="./eval_data", type=str, help="Path to the DTU evaluation data directory.")
parse.add_argument("-s", "--scan", default=1, type=int, help="The scan number being evaluated.")
ARGS = parse.parse_args()

def main():
    # hard coded for now...
    s = 1
    network="gbinet"
    scan = "scan{:03d}".format(s)
    gt_depth_dir = "/media/nate/Data/Fusion/dtu/{}/GT_Depths/{}/".format(network, scan)
    input_depth_dir = "/media/nate/Data/Results/V-FUSE/dtu/Output_{}/{}/input_depths/".format(network, scan)
    fused_depth_dir = "/media/nate/Data/Results/V-FUSE/dtu/Output_{}/{}/depths/".format(network, scan)
    camera_dir = "/media/nate/Data/Results/V-FUSE/dtu/Output_{}/{}/cams/".format(network, scan)
    output_dir = "/media/nate/Data/Results/V-FUSE/dtu/Output_{}/stats/{}/".format(network, scan)
    eval_data_dir = "/media/nate/Data/Evaluation/dtu/mvs_data/"

    # extract arguments
    #   scan = "scan{:03d}".format(ARGS.scan)
    #   gt_depth_dir = ARGS.gt_depth_dir
    #   input_depth_dir = ARGS.input_depth_dir
    #   fused_depth_dir = ARGS.fused_depth_dir
    #   camera_dir = ARGS.camera_dir
    #   output_dir = ARGS.output_dir
    #   eval_data_dir = ARGS.eval_data_dir

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

    cam_files = os.listdir(camera_dir)
    cam_files.sort()
    cam_files = [os.path.join(camera_dir,c) for c in cam_files if c[-4:]==".txt"]

    total_views = len(fused_depth_files)

    # build view-colored point clouds
    total_iae = np.zeros(total_views-1)
    total_fae = np.zeros(total_views-1)
    total_density = np.zeros(total_views-1)
    view_vecs = np.zeros((total_views, 3))
    with tqdm(fused_depth_files, unit="views") as data_loader:
        for view_num,fdf in enumerate(data_loader):
            # get file names
            gdf = gt_depth_files[view_num]
            idf = input_depth_files[view_num]
            fdf = fused_depth_files[view_num]
            cf = cam_files[view_num]

            # load data
            gt_depth = read_pfm(gdf)
            input_depth = read_pfm(idf)
            fused_depth = read_pfm(fdf)
            cam = read_cam(open(cf,'r'))

            # compute depth map errors
            (iae, fae) = abs_error(input_depth, fused_depth, gt_depth)

            # compute visibility scores
            gt_vis_map = visibility(gt_depth, cam, gt_depth_files, cam_files, view_num)

            # flatten matrices and remove any point that does not correspond to a gt depth
            iae = iae.flatten()
            fae = fae.flatten()
            gt_vis_map = gt_vis_map.flatten()
            gt_depth = gt_depth.flatten()
            iae = [ i for ind,i in enumerate(iae) if gt_depth[ind] != 0 ]
            fae = [ f for ind,f in enumerate(fae) if gt_depth[ind] != 0 ]
            gt_vis_map = [ g for ind,g in enumerate(gt_vis_map) if gt_depth[ind] != 0 ]

            avg_iae = np.zeros(total_views-1)
            avg_fae = np.zeros(total_views-1)
            avg_density = np.zeros(total_views-1)
            views = list(range(1,total_views))
            # average based on visibility score
            for v in views:
                v_mask = np.where(np.equal(gt_vis_map, v))[0]

                iae_v = np.take(iae, v_mask)
                fae_v = np.take(fae, v_mask)
                avg_density[v-1] = v_mask.shape[0]

                if (v_mask.shape[0] == 0):
                    avg_iae[v-1] = 0.0
                    avg_fae[v-1] = 0.0
                else:
                    avg_iae[v-1] = np.mean(iae_v)
                    avg_fae[v-1] = np.mean(fae_v)
            avg_density = avg_density / (np.max(avg_density))

            # plot per-view errors vs visibility
            plt.plot(views, avg_iae, label="input")
            plt.plot(views, avg_fae, label="fused")
            plt.legend()
            plt.xlabel("visibility")
            plt.ylabel("absolute error")
            plt.savefig(os.path.join(output_dir,"{:04d}_visibility_2d.png".format(view_num)), dpi=300)
            plt.close()

            # plot per-view density
            plt.bar(views, avg_density)
            plt.xlabel("visibility")
            plt.savefig(os.path.join(output_dir,"{:04d}_density.png".format(view_num)), dpi=300)
            plt.close()
             

            #   # plot per-scene errors vs visibility
            total_iae += avg_iae
            total_fae += avg_fae
            plt.plot(views, total_iae/(view_num+1), label="input")
            plt.plot(views, total_fae/(view_num+1), label="fused")
            plt.legend()
            plt.xlabel("visibility")
            plt.ylabel("absolute error")
            plt.savefig(os.path.join(output_dir,"visibility_total_2d.png"), dpi=300)
            plt.close()

            # plot per-scene density
            total_density += avg_density
            plt.bar(views, total_density/(view_num+1))
            plt.xlabel("visibility")
            plt.savefig(os.path.join(output_dir,"density_total.png"), dpi=300)
            plt.close()
    
if __name__=="__main__":
    main()
