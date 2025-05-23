# cvt/camera.py

"""A suite of common camera utilities.

This module includes several functions for manipulating and extracting information
from camera intrinsics and extrinsics, as well as converting between specific
formats.
"""

import os
import numpy as np
import math
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation
import torch
import open3d as o3d
from typing import Tuple


def build_o3d_traj(poses: np.ndarray, K: np.ndarray, width: int, height: int) -> o3d.camera.PinholeCameraTrajectory:
    """Converts an array of camera poses and an intrinsics matrix into Open3D trajectory format.

    Parameters:
        poses: Nx4x4 array of camera pose in world-to-camera format.
        K: 3x3 camera intrinsics matrix.
        width: Expected image width.
        height: Expected image height.

    Returns:
        Open3D camera trajectory model.
    """
    trajectory = o3d.camera.PinholeCameraTrajectory()
    for pose in poses:
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
        camera_params.extrinsic = pose
        trajectory.parameters += [(camera_params)]
    return trajectory

def camera_center_(pose: np.ndarray) -> np.ndarray:
    """Computes the center of a camera in world coordinates.

    Parameters:
        pose: The extrinsics matrix (4x4) of a given camera in world-to-camera format.

    Returns:
        The camera center vector (3x1) in world coordinates.
    """
    C = null_space(pose[:3,:4])
    C /= C[3,:]

    return C[:3,0]

def compute_baselines_(poses: np.ndarray) -> Tuple[float, float]:
    """Computes the minimum and maximum baseline between a reference camera pose (poses[0]) and a cluster of supporting camera poses (poses[1:]).

    Parameters:
        poses: Array of camera poses (Nx4x4)

    Returns:
        A tuple of baselines where the first element is the minimum baseline and the second element is the maximum baseline.
    """
    ref_pose = poses[0]
    ref_camera_center = null_space(ref_pose[:3,:])
    ref_camera_center = ref_camera_center[:3,0] / ref_camera_center[3,0]
    min_baseline = np.inf
    max_baseline = 0.0

    num_views = len(poses)
    for i in range(1, num_views):
        sup_pose = poses[i]
        sup_camera_center = null_space(sup_pose[:3,:])
        sup_camera_center = sup_camera_center[:3,0] / sup_camera_center[3,0]
        b = np.linalg.norm(ref_camera_center - sup_camera_center)
        if b < min_baseline:
            min_baseline = b
        if b > max_baseline:
            max_baseline = b
    return min_baseline, max_baseline


def crop_intrinsics_(K: np.ndarray, crop_row: int, crop_col: int) -> np.ndarray:
    """Adjusts intrinsics matrix principle point coordinates corresponding to image cropping.

    Parameters:
        K: 3x3 camera intrinsics matrix.
        crop_row: Offset for cy (corresponding to cropping the left and right side of an image by 'crop_row').
        crop_col: Offset for cx (corresponding to cropping the top and bottom side of an image by 'crop_col').

    Returns:
        The updated 3x3 intrinsics matrix
    """

    K[0,2] -= crop_col
    K[1,2] -= crop_row
    return K


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def intrinsic_pyramid(K: torch.tensor, levels: int) -> torch.tensor:
    """Computes camera intrinsics pyramid corresonding to several levels of image downsampling.

    Parameters:
        K: [Batch x 3 x 3] camera intrinsics matrix.
        levels: The number of pyramid levels to compute.

    Returns:
        The intrinsics pyramid of shape [Batch x L x 3 x 3].
    """
    batch_size, _, _ = K.shape
    K_pyr = torch.zeros((batch_size, levels, 3, 3)).to(K)

    for l in range(levels):
        if l==0:
            k = K
        else:
            k = torch.clone(K) / (2**l)
            k[:,2,2] = 1.0
        K_pyr[:,l] = k

    return K_pyr


def intrinsic_pyramid_(K: np.ndarray, levels: int) -> np.ndarray:
    """Computes camera intrinsics pyramid corresonding to several levels of image downsampling.

    Parameters:
        K: [3 x 3] camera intrinsics matrix.
        levels: The number of pyramid levels to compute.

    Returns:
        The intrinsics pyramid of shape [L x 3 x 3].
    """
    K_pyr = np.zeros((levels, 3, 3))

    for l in range(levels):
        if l==0:
            k = K
        else:
            k = np.copy(K) / (2**l)
            k[2,2] = 1.0
        K_pyr[l] = k

    return K_pyr


def relative_transform_(poses_1: np.ndarray, poses_2: np.ndarray) -> np.ndarray:
    """Computes the approximate relative transformation between two sets of camera trajectories.

    Args:
        poses_1: Array of the first set of cameras (Nx4x4).
        poses_2: Array of the second set of cameras (Nx4x4).

    Returns:
        The relative transformation matrix (4x4) between the two trajectories.
    """
    centers_1 = np.squeeze(np.array([ camera_center(p) for p in poses_1 ]), axis=2)
    centers_2 = np.squeeze(np.array([ camera_center(p) for p in poses_2 ]), axis=2)

    ### determine scale
    # grab first camera pair
    c1_0 = centers_1[0][:3]
    c2_0 = centers_2[0][:3]

    # grab last camera pair
    c1_1 = centers_1[-1][:3]
    c2_1 = centers_2[-1][:3]

    # calculate the baseline between both sets of cameras
    baseline_1 = np.linalg.norm(c1_0 - c1_1)
    baseline_2 = np.linalg.norm(c2_0 - c2_1)

    # compute the scale based on the baseline ratio
    scale = baseline_2/baseline_1

    ### determine 1->2 Rotation 
    b1 = np.array([c[:3] for c in centers_1])
    b2 = np.array([c[:3] for c in centers_2])
    R = Rotation.align_vectors(b2,b1)[0].as_matrix()
    R = scale * R

    ### create transformation matrix
    M = np.eye(4)
    M[:3,:3] = R

    ### determine 1->2 Translation
    num_cams = len(poses_1)
    t = np.array([ c2-(M@c1) for c1,c2 in zip(centers_1,centers_2) ])
    t = np.mean(t, axis=0)

    ### add translation
    M[:3,3] = t[:3]

    return M

def scale_intrinsics(K: torch.tensor, scale: float) -> torch.tensor:
    """Adjust intrinsics matrix focal length and principle point corresponding to image scaling.

    Parameters:
        K: Batchx3x3 camera intrinsics matrices.
        scale: Scale amount (i.e. scale=0.5 corresponds to scaling an image by a factor of 1/2).

    Returns:
        The updated 3x3 intrinsics matrix.
    """
    K_clone = torch.clone(K)
    K_clone[:,:2,:] = K_clone[:,:2,:] * scale

    return K_clone

def scale_intrinsics_(K: np.ndarray, scale: float) -> np.ndarray :
    """Adjust intrinsics matrix focal length and principle point corresponding to image scaling.

    Parameters:
        K: 3x3 camera intrinsics matrix.
        scale: Scale amount (i.e. scale=0.5 corresponds to scaling an image by a factor of 1/2).


    Returns:
        The updated 3x3 intrinsics matrix.
    """
    K_copy = K.copy()
    K_copy[:2, :] = K_copy[:2, :] * scale

    return K_copy

def sfm_to_trajectory(cams: np.ndarray, log_file: str) -> None:
    """Convert a set of cameras from SFM format to Trajectory File format.

    Parameters:
        cams: Array of camera extrinsics (Nx4x4) to be converted.
        log_file: Output path to the *.log file that is to be created.
    """
    num_cams = len(cams)

    with open(log_file, 'w') as f:
        for i,cam in enumerate(cams):
            # write camera to output_file
            f.write("{} {} 0\n".format(str(i),str(i)))
            for row in cam:
                for c in row:
                    f.write("{} ".format(str(c)))
                f.write("\n")
        
    return

def to_opengl_pose(pose: torch.tensor) -> torch.tensor:
    """OpenGL pose: (right-up-back) (cam-to-world)

    Parameters:
        pose: The pose in non-OpenGL format

    Returns:
        Camera pose in OpenGL format.
    """
    pose = torch.linalg.inv(pose)
    pose[:3,1] *= -1
    pose[:3,2] *= -1
    return pose

def to_opengl_pose_(pose: np.ndarray) -> np.ndarray:
    """OpenGL pose: (right-up-back) (cam-to-world)

    Parameters:
        pose: The pose in non-OpenGL format

    Returns:
        Camera pose in OpenGL format.
    """
    pose = np.linalg.inv(pose)
    pose[:3,1] *= -1
    pose[:3,2] *= -1
    return pose

def trajectory_to_sfm(log_file: str, camera_path: str, intrinsics: np.ndarray) -> None:
    """Convert a set of cameras from Trajectory File format to SFM format.

    Args:
        log_file: Input *.log file that stores the trajectory information.
        camera_path: Output path where the SFM camera files will be written.
        intrinsics: Array of intrinsics matrices (Nx3x3) for each camera.
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
        num_lines = len(lines)
        i = 0

        while(i < num_lines-5):
            view_num = int(lines[i].strip().split(" ")[0])
            
            cam = np.zeros((2,4,4))
            cam[0,0,:] = np.asarray(lines[i+1].strip().split(" "), dtype=float)
            cam[0,1,:] = np.asarray(lines[i+2].strip().split(" "), dtype=float)
            cam[0,2,:] = np.asarray(lines[i+3].strip().split(" "), dtype=float)
            cam[0,3,:] = np.asarray(lines[i+4].strip().split(" "), dtype=float)
            cam[0,:,:] = np.linalg.inv(cam[0,:,:])

            cam_file = "{:08d}_cam.txt".format(view_num)
            cam_path = os.path.join(camera_path, cam_file)

            cam[1,:,:] = intrinsics[view_num]

            write_cam(cam_path, cam)
            i = i+5
    return

def y_axis_rotation(P: np.ndarray, theta: float) -> np.ndarray:
    """Applies a rotation to the given camera extrinsics matrix along the y-axis.

    Parameters:
        P: Initial extrinsics camera matrix.
        theta: Angle (in radians) to rotate the camera.

    Returns:
        The rotated extrinsics matrix for the camera.
    """
    R = np.eye(4)
    R[0,0] = math.cos(theta)
    R[0,2] = math.sin(theta)
    R[2,0] = -(math.sin(theta))
    R[2,2] = math.cos(theta)

    P_rot = R @ P

    return P_rot

def z_planes_from_disp(Z: torch.tensor, b: torch.tensor, f: torch.tensor, delta: float) -> Tuple[torch.tensor, torch.tensor]:
    """Computes the near and far Z planes corresponding to 'delta' disparity steps between two cameras.

    Parameters:
        Z: Z buffer storing D depth plane hypotheses [B x C x D x H x W]. (shape resembles a typical PSV).
        b: The baseline between cameras [B].
        f: The focal length of camera [B].
        delta: The disparity delta for the near and far planes.

    Returns:
        The tuple of near and far Z planes corresponding to 'delta' disparity steps.
    """
    B, C, D, H, W = Z.shape
    b = b.reshape(B,1,1,1,1).repeat(1,C,D,H,W)
    f = f.reshape(B,1,1,1,1).repeat(1,C,D,H,W)

    near = Z*(b*f/((b*f) + (delta*Z)))
    far = Z*(b*f/((b*f) - (delta*Z)))

    return (near, far)
