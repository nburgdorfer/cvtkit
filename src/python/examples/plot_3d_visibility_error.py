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
            cam = read_cam(open(cf,'r'))

            # build per-view ply (points encoded with view number)
            view_vecs[view_num] = np.asarray([(view_num/total_views), 0, 1-(view_num/total_views)])
            input_view_ply = build_cloud(input_depth, cam, view_vecs[view_num])
            input_ply += input_view_ply

            fused_view_ply = build_cloud(fused_depth, cam, view_vecs[view_num])
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
    print("Computing accuracy...")
    input_acc_ply, input_acc_dists, input_acc_colors = accuracy_eval(input_ply, gt_ply, 20, 0.4, 0.0, input_filt, gt_filt)
    fused_acc_ply, fused_acc_dists, fused_acc_colors = accuracy_eval(fused_ply, gt_ply, 20, 0.4, 0.0, fused_filt, gt_filt)

    print("Computing completeness...")
    input_comp_points, input_comp_dists, input_comp_colors = completeness_eval(input_ply, gt_ply, 20, 0.4, 0.0, input_filt, gt_filt)
    fused_comp_points, fused_comp_dists, fused_comp_colors = completeness_eval(fused_ply, gt_ply, 20, 0.4, 0.0, fused_filt, gt_filt)


    # compute visibility statistics
    total_iam = np.zeros(total_views-1)
    total_fam = np.zeros(total_views-1)
    total_icm = np.zeros(total_views-1)
    total_fcm = np.zeros(total_views-1)
    total_density = np.zeros(total_views-1)
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

            # get projected view acc and comp
            input_view_inds = np.where(np.all(input_acc_colors == view_vecs[view_num], axis=1))[0]
            input_acc_map = project_2d( \
                                np.asarray(input_acc_ply.select_by_index(input_view_inds).points), \
                                input_acc_dists[input_view_inds], \
                                input_depth.shape, \
                                cam)
            input_view_inds = np.where(np.all(input_comp_colors == view_vecs[view_num], axis=1))[0]
            input_comp_map = project_2d( \
                                np.squeeze(input_comp_points[input_view_inds], axis=1), \
                                input_comp_dists[input_view_inds], \
                                input_depth.shape, \
                                cam)

            # get projected view acc and comp
            fused_view_inds = np.where(np.all(fused_acc_colors == view_vecs[view_num], axis=1))[0]
            fused_acc_map = project_2d( \
                                np.asarray(fused_acc_ply.select_by_index(fused_view_inds).points), \
                                fused_acc_dists[fused_view_inds], \
                                fused_depth.shape, \
                                cam)
            fused_view_inds = np.where(np.all(fused_comp_colors == view_vecs[view_num], axis=1))[0]
            fused_comp_map = project_2d( \
                                np.squeeze(fused_comp_points[fused_view_inds], axis=1), \
                                fused_comp_dists[fused_view_inds], \
                                fused_depth.shape, \
                                cam)

            # compute visibility scores
            gt_vis_map = visibility(gt_depth, cam, gt_depth_files, cam_files, view_num)

            # flatten matrices and remove any point that does not correspond to a gt depth
            iam = input_acc_map.flatten()
            icm = input_comp_map.flatten()
            fam = fused_acc_map.flatten()
            fcm = fused_comp_map.flatten()
            gt_vis_map = gt_vis_map.flatten()
            gt_depth = gt_depth.flatten()
            iam = [ ia for ind,ia in enumerate(iam) if gt_depth[ind] != 0 ]
            icm = [ ic for ind,ic in enumerate(icm) if gt_depth[ind] != 0 ]
            fam = [ fa for ind,fa in enumerate(fam) if gt_depth[ind] != 0 ]
            fcm = [ fc for ind,fc in enumerate(fcm) if gt_depth[ind] != 0 ]
            gt_vis_map = [ g for ind,g in enumerate(gt_vis_map) if gt_depth[ind] != 0 ]

            avg_iam = np.zeros(total_views-1)
            avg_icm = np.zeros(total_views-1)
            avg_fam = np.zeros(total_views-1)
            avg_fcm = np.zeros(total_views-1)
            avg_density = np.zeros(total_views-1)
            views = list(range(1,total_views))
            # average based on visibility score
            for v in views:
                v_mask = np.where(np.equal(gt_vis_map, v))[0]

                iam_v = np.take(iam, v_mask)
                icm_v = np.take(icm, v_mask)
                fam_v = np.take(fam, v_mask)
                fcm_v = np.take(fcm, v_mask)
                avg_density[v-1] = v_mask.shape[0]

                if (v_mask.shape[0] == 0):
                    avg_iam[v-1] = 0.0
                    avg_icm[v-1] = 0.0
                    avg_fam[v-1] = 0.0
                    avg_fcm[v-1] = 0.0
                else:
                    avg_iam[v-1] = np.mean(iam_v)
                    avg_icm[v-1] = np.mean(icm_v)
                    avg_fam[v-1] = np.mean(fam_v)
                    avg_fcm[v-1] = np.mean(fcm_v)
            avg_density = avg_density / (np.max(avg_density))

            # plot per-view errors vs visibility
            plt.plot(views, avg_iam, label="input_acc")
            plt.plot(views, avg_fam, label="fused_acc")
            plt.plot(views, avg_icm, label="input_comp")
            plt.plot(views, avg_fcm, label="fused_comp")
            plt.legend()
            plt.xlabel("visibility")
            plt.ylabel("absolute error")
            plt.savefig(os.path.join(output_dir,"{:04d}_visibility_3d.png".format(view_num)), dpi=300)
            plt.close()

            # plot per-view density
            plt.bar(views, avg_density)
            plt.xlabel("visibility")
            plt.savefig(os.path.join(output_dir,"{:04d}_density.png".format(view_num)), dpi=300)
            plt.close()
             

            # plot per-scene errors vs visibility
            total_iam += avg_iam
            total_fam += avg_fam
            total_icm += avg_icm
            total_fcm += avg_fcm

            plt.plot(views, total_iam/(view_num+1), label="input_acc")
            plt.plot(views, total_fam/(view_num+1), label="fused_acc")
            plt.plot(views, total_icm/(view_num+1), label="input_comp")
            plt.plot(views, total_fcm/(view_num+1), label="fused_comp")
            plt.legend()
            plt.xlabel("visibility")
            plt.ylabel("absolute error")
            plt.savefig(os.path.join(output_dir,"visibility_total_3d.png"), dpi=300)
            plt.close()

            # plot per-scene density
            total_density += avg_density
            plt.bar(views, total_density/(view_num+1))
            plt.xlabel("visibility")
            plt.savefig(os.path.join(output_dir,"density_total.png"), dpi=300)
            plt.close()

if __name__=="__main__":
    main()
