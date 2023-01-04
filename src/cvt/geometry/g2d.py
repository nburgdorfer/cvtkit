# geometry/ g2d.py

"""Module including routines based in 2D geometry.

This 2D geometry module contains functions that

This module contains the following functions:

- `match_features(query_img, train_img, max_features)` - .
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import os
import sys
from typing import Tuple

from common.io import *

def match_features(query_img: np.ndarray, train_img: np.ndarray, max_features: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Computer matching ORB features between a pair of images.

    Args:
        query_img: The first of a pair of images to compute and match features.
        train_img: The second of a pair of images to compute and match features.
        max_features: The maximum number of features to retain.

    Returns:
        pts1: The set of matched point coordinates for the first image.
        pts2: The set of matched point coordinates for the second image.
    """
    query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
      
    orb = cv2.ORB_create(max_features)
      
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = list(matcher.match(queryDescriptors,trainDescriptors) )
    matches.sort(key = lambda x:x.distance)

    pts1 = []
    pts2 = []
    for i in range(8):
        m = matches[i]

        pts2.append(trainKeypoints[m.trainIdx].pt)
        pts1.append(queryKeypoints[m.queryIdx].pt)
    pts1  = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    return (pts1, pts2)

def compute_homography(img1_filename, img2_filename):
    img1 = cv2.imread(img1_filename)
    img2 = cv2.imread(img2_filename)

    (height, width, _) = img1.shape

    (pts1, pts2) = match_features(img1, img2)

    # Compute fundamental matrix
    H, mask = cv2.findHomography(pts2, pts1, method=cv2.RANSAC)

    #warped_img = np.zeros((height,width))
    #warped_img = cv2.warpPerspective(src=img2, M=H, dsize=(width,height))
    #cv2.imwrite("data/warped.png", warped_img)

    return H

def compute_essential_matrix(img1_filename, img2_filename, K):
    img1 = cv2.imread(img1_filename)
    img2 = cv2.imread(img2_filename)

    # compute matching features
    (pts1, pts2) = match_features(img1, img2)

    # Compute fundamental matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)

    # decompose into rotation / translation
    R1, R2, t = cv2.decomposeEssentialMat(E)
    print(R1)
    print(R2)
    print(t)

    return E

def fundamental_from_features(img1_filename, img2_filename):
    img1 = cv2.imread(img1_filename)
    img2 = cv2.imread(img2_filename)

    # compute matching features
    (pts1, pts2) = match_features(img1, img2)
    
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
    return F

def fundamentalFromKP(K,P1,P2) :
    R1 = P1[0:3,0:3]
    t1 = P1[0:3,3]
    R2 = P2[0:3,0:3]
    t2 = P2[0:3,3]

    t1aug = np.array([t1[0], t1[1], t1[2], 1])
    epi2 = np.matmul(P2,t1aug)
    epi2 = np.matmul(K,epi2[0:3])
    print('epipole 2: {} {}'.format(epi2[0]/epi2[2],epi2[1]/epi2[2]))

    R = np.matmul(R2,np.transpose(R1))
    t= t2- np.matmul(R,t1)
    K1inv = np.linalg.inv(K)
    K2invT = np.transpose(K1inv)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.matmul(K2invT,np.matmul(tx,np.matmul(R,K1inv)))
    F = F/np.amax(F)
    return F



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

    return vis_map.astype(np.float32)

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
