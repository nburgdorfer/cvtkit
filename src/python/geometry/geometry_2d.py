import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import os
import sys

from common.io import *

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
        src_depth = read_pfm(sdf)
        src_cam = read_cam(open(scf,'r'))

        mask = check_geometric_consistency(ref_depth, ref_cam, src_depth, src_cam, pix_th)
        vis_map += mask

    return vis_map

def project_ply(ply, dists, cam, width, height):

    cmap = plt.get_cmap("hot_r")
    colors = cmap(dists)[:, :3]
    ply.colors = o3d.utility.Vector3dVector(colors)

    # set up the renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    render.scene.add_geometry("cloud", ply, mat)
    render.scene.set_background(np.asarray([0,0,0,1])) #r,g,b,a
    intrins = o3d.camera.PinholeCameraIntrinsic(width, height, cam[1,0,0], cam[1,1,1], cam[1,0,2], cam[1,1,2])
    render.setup_camera(intrins, cam[0])
    image = np.asarray(render.render_to_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image    

def build_cloud(depth, cam, view_vec):
    ply = o3d.geometry.PointCloud()

    # extract camera params
    height, width = depth.shape
    fx = cam[1,0,0]
    fy = cam[1,1,1]
    cx = cam[1,0,2]
    cy = cam[1,1,2]
    intrins = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    extrins = cam[0]

    # convert deth to o3d.geometry.Image
    depth_map = o3d.geometry.Image(depth)

    ply = ply.create_from_depth_image(depth_map, intrins, extrins, depth_scale=1.0, depth_trunc=1000)
    view_colors = o3d.utility.Vector3dVector(np.full((len(ply.points), 3), view_vec))

    ply.colors = view_colors
    
    return ply
