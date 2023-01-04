# common/camera.py

"""A suite of common camera utilities.

This module includes several functions for manipulating and extracting information
from camera intrinsics and extrinsics, as well as converting between specific
formats.

This module contains the following functions:

- `camera_center(cam)` - Computes the center of a camera in world coordinates.
- `relative_transform(cams_1, cams_2)` - Computes the relative transformation between two sets of cameras.
- `convert_to_log(cams, output_file, alignment)` - Converts from DTU camera format to log format.
- `convert_from_log(log_file, output_path, old_cam_path)` - Converts from log camera format to DTU format.
"""

import os
import sys
import numpy as np
import cv2
import re
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation

def camera_center(cam: np.ndarray) -> np.ndarray:
    """Computes the center of a camera in world coordinates.

    Args:
        cam: The extrinsics matrix (4x4) of a given camera.

    Returns:
        The camera center vector (3x1) in world coordinates.
    """
    C = null_space(cam[:3,:4])
    C /= C[3,:]

    return C

def relative_transform(cams_1: np.ndarray, cams_2: np.ndarray) -> np.ndarray:
    """Computes the relative transformation between two sets of cameras.

    Args:
        cams_1: Array of the first set of cameras (Nx4x4).
        cams_2: Array of the second set of cameras (Nx4x4).

    Returns:
        The relative transformation matrix (4x4) between the two trajectories.
    """
    centers_1 = np.squeeze(np.array([ camera_center(c) for c in cams_1 ]), axis=2)
    centers_2 = np.squeeze(np.array([ camera_center(c) for c in cams_2 ]), axis=2)

    ### determine scale
    # grab first camera pair
    c1_0 = centers_1[0][:3]
    c2_0 = centers_2[0][:3]

    # grab one-hundreth camera pair
    c1_1 = centers_1[99][:3]
    c2_1 = centers_2[99][:3]

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
    num_cams = len(cams_1)
    t = np.array([ c2-(M@c1) for c1,c2 in zip(centers_1,centers_2) ])
    t = np.mean(t, axis=0)

    ### add translation
    M[:3,3] = t[:3]

    return M


def sfm_to_trajectory(cams: np.ndarray, log_file: str) -> None:
    """Convert a set of cameras from SFM format to Trajectory File format.

    Args:
        cams: Array of camera extrinsics (Nx4x4) to be converted.
        log_file: Output path to the *.log file that is to be created.
    """
    num_cams = len(cams)

    with open(output_file, 'w') as f:
        for i,cam in enumerate(cams):
            # write camera to output_file
            f.write("{} {} 0\n".format(str(i),str(i)))
            for row in cam:
                for c in row:
                    f.write("{} ".format(str(c)))
                f.write("\n")
        
    return

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
