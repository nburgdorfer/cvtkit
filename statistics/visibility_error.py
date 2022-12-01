import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# custom imports
sys.path.append("../common_utilities")
from utils import *


def abs_error(input_depth, fused_depth, gt_depth):
    input_signed_error = input_depth - gt_depth
    fused_signed_error = fused_depth - gt_depth

    # compute gt mask and number of valid pixels
    gt_mask = np.not_equal(gt_depth, 0.0).astype(np.double)
    input_abs_error = np.abs(input_signed_error) * gt_mask
    fused_abs_error = np.abs(fused_signed_error) * gt_mask

    return input_abs_error, fused_abs_error


def reproject_with_depth(ref_depth, ref_cam, src_depth, src_cam):
    height, width = ref_depth.shape

    # back-project ref depths to 3D
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    xyz_ref = np.matmul(np.linalg.inv(ref_cam[1,:3,:3]),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * ref_depth.reshape([-1]))

    # transform 3D points from ref to src coords
    xyz_src = np.matmul(np.matmul(src_cam[0], np.linalg.inv(ref_cam[0])),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]

    # project src 3D points into pixel coords
    K_xyz_src = np.matmul(src_cam[1,:3,:3], xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)

    # sample the depth values from the src map at each pixel coord
    sampled_depth_src = cv2.remap(src_depth, x_src, y_src, interpolation=cv2.INTER_LINEAR)

    # back-project src depths to 3D
    xyz_src = np.matmul(np.linalg.inv(src_cam[1,:3,:3]),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))

    # transform 3D points from src to ref coords
    xyz_reprojected = np.matmul(np.matmul(ref_cam[0], np.linalg.inv(src_cam[0])),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]

    # extract reprojected depth values
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)

    # project ref 3D points into pixel coords
    K_xyz_reprojected = np.matmul(ref_cam[1,:3,:3], xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-7)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(ref_depth, ref_cam, src_depth, src_cam, pix_th):
    height, width = ref_depth.shape
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))

    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(ref_depth, ref_cam, src_depth, src_cam)

    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
    mask = np.where(dist < pix_th, 1, 0)

    return mask


def visibility(ref_depth, ref_cam, depth_files, cam_files, ref_ind, pix_th=0.1):
    height, width = ref_depth.shape
    vis_map = np.not_equal(ref_depth, 0.0).astype(np.double)

    for i in range(len(depth_files)):
        if (i==ref_ind):
            continue

        # get files
        sdf = depth_files[i]
        scf = cam_files[i]

        # load data
        src_depth = load_pfm(open(sdf,'rb'))
        src_cam = load_cam(open(scf,'r'))

        mask = check_geometric_consistency(ref_depth, ref_cam, src_depth, src_cam, pix_th)
        vis_map += mask

    return vis_map


def main():
    scans = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
    network="gbinet"

    for s in scans:
        scan = "scan{:03d}".format(s)
        gt_depth_path = "/media/nate/Data/Fusion/dtu/{}/GT_Depths/{}/".format(network, scan)
        input_depth_path = "/media/nate/Data/Results/V-FUSE/dtu/Output_{}/{}/input_depths/".format(network, scan)
        fused_depth_path = "/media/nate/Data/Results/V-FUSE/dtu/Output_{}/{}/depths/".format(network, scan)
        cam_path = "/media/nate/Data/Results/V-FUSE/dtu/Output_{}/{}/cams/".format(network, scan)
        output_path = "/media/nate/Data/Results/V-FUSE/dtu/Output_{}/stats/{}/".format(network, scan)

        

        gt_depth_files = os.listdir(gt_depth_path)
        gt_depth_files.sort()
        gt_depth_files = [os.path.join(gt_depth_path,g) for g in gt_depth_files if g[-9:]=="depth.pfm"]

        input_depth_files = os.listdir(input_depth_path)
        input_depth_files.sort()
        input_depth_files = [os.path.join(input_depth_path,i) for i in input_depth_files if i[-4:]==".pfm"]

        fused_depth_files = os.listdir(fused_depth_path)
        fused_depth_files.sort()
        fused_depth_files = [os.path.join(fused_depth_path,f) for f in fused_depth_files if f[-4:]==".pfm"]

        cam_files = os.listdir(cam_path)
        cam_files.sort()
        cam_files = [os.path.join(cam_path,c) for c in cam_files if c[-4:]==".txt"]

        num_views = len(gt_depth_files)

        total_iae = np.zeros(num_views-1)
        total_fae = np.zeros(num_views-1)


        with tqdm(gt_depth_files, unit="views") as data_loader:
            for view_num,gdf in enumerate(data_loader):
                # get file names
                idf = input_depth_files[view_num]
                fdf = fused_depth_files[view_num]
                cf = cam_files[view_num]

                # load data
                gt_depth = load_pfm(open(gdf,'rb'))
                input_depth = load_pfm(open(idf,'rb'))
                fused_depth = load_pfm(open(fdf,'rb'))
                cam = load_cam(open(cf,'r'))

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

                avg_iae = np.zeros(num_views-1)
                avg_fae = np.zeros(num_views-1)
                views = list(range(1,num_views))
                # average based on visibility score
                for v in views:
                    v_mask = np.where(np.equal(gt_vis_map, v))[0]
                    iae_v = np.take(iae, v_mask)
                    fae_v = np.take(fae, v_mask)

                    if (v_mask.shape[0] == 0):
                        avg_iae[v-1] = 0.0
                        avg_fae[v-1] = 0.0
                    else:
                        avg_iae[v-1] = np.mean(iae_v)
                        avg_fae[v-1] = np.mean(fae_v)

                plt.plot(views, avg_iae, label="input")
                plt.plot(views, avg_fae, label="fused")
                plt.legend()
                plt.xlabel("visibility")
                plt.ylabel("absolute error")
                plt.savefig(os.path.join(output_path,"{:04d}_visibility.png".format(view_num)), dpi=300)
                plt.close()

                total_iae += avg_iae
                total_fae += avg_fae
                plt.plot(views, total_iae/(view_num+1), label="input")
                plt.plot(views, total_fae/(view_num+1), label="fused")
                plt.legend()
                plt.xlabel("visibility")
                plt.ylabel("absolute error")
                plt.savefig(os.path.join(output_path,"visibility_total.png"), dpi=300)
                plt.close()

        old_path = os.path.join(output_path,"visibility_total.png")
        new_path = os.path.join("/media/nate/Data/Results/V-FUSE/dtu/Output_{}/stats/".format(network),"{:03d}_visibility_total.png".format(s))
        
        shutil.copy(old_path, new_path)

            
if __name__=="__main__":
    main()
