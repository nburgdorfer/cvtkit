# cvt/io.py

"""A suite of common input/output functions.

This module includes several functions for reading and writing
different types of data useful for computer vision applications.

This module contains the following functions:

- `read_cams_sfm(camera_path, extension)` - Reads an entire directory of camera files in SFM format.
- `read_cams_trajectory(log_file)` - Reads camera file in Trajectory File format.
- `read_matrix(mat_file)` - Reads a single matrix of float values from a file.
- `read_mesh(mesh_file)` - Reads a mesh from a file.
- `read_pfm(pfm_file)` - Reads a file in *.pfm format.
- `read_point_cloud(point_cloud_file)` - Reads a point cloud from a file.
- `read_single_cam_sfm(cam_file, depth_planes)` - Reads a single camera file in SFM format.
- `read_stereo_intrinsics_yaml(intrinsics_file)` - Reads intrinsics information for a stereo camera pair from a *.yaml file.
- `write_matrix(M, mat_file)` - Writes a single matrix to a file.
- `write_mesh(mesh_file, mesh)` - Writes a mesh to a file.
- `write_pfm(pfm_file, data_map, scale)` - Writes a data map to a file in *.pfm format.
"""

import os
import sys
import numpy as np
import cv2
import re
import open3d as o3d
from typing import List, Tuple


def read_pfm(pfm_file: str) -> np.ndarray:
    """Reads a file in *.pfm format.

    Args:
        pfm_file: Input *.pfm file to be read.

    Returns:
        Data map that was stored in the *.pfm file.
    """
    with open(pfm_file, 'rb') as pfm_file:
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = pfm_file.readline().decode('iso8859_15').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('iso8859_15'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        # scale = float(file.readline().rstrip())
        scale = float((pfm_file.readline()).decode('iso8859_15').rstrip())
        if scale < 0: # little-endian
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = pfm_file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
    return data

def write_pfm(pfm_file: str, data_map: np.ndarray, scale: float = 1.0) -> None:
    """Writes a data map to a file in *.pfm format.

    Args:
        pfm_file: Output *.pfm file to store the data map.
        data_map: Data map to be stored.
        scale: Value used to scale the data map.
    """
    with open(pfm_file, 'wb') as pfm_file:
        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3: # color image
            color = True
        elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1): # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        a = 'PF\n' if color else 'Pf\n'
        b = '%d %d\n' % (image.shape[1], image.shape[0])
        
        pfm_file.write(a.encode('iso8859-15'))
        pfm_file.write(b.encode('iso8859-15'))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        c = '%f\n' % scale
        pfm_file.write(c.encode('iso8859-15'))

        image_string = image.tostring()
        pfm_file.write(image_string)


def read_single_cam_sfm(cam_file: str, depth_planes: int = 256) -> np.ndarray:
    """Reads a single camera file in SFM format.

    Args:
        cam_file: Input camera file to be read.
        depth_planes: Number of depth planes to store in the view metadata.

    Returns:
        Camera extrinsics, intrinsics, and view metadata (2x4x4).
    """
    cam = np.zeros((2, 4, 4))
    words = cam_file.read().split()
    words_len = len(words)

    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0,i,j] = float(words[extrinsic_index])

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1,i,j] = float(words[intrinsic_index])

    if words_len == 29:
        cam[1,3,0] = float(words[27])
        cam[1,3,1] = float(words[28])
        cam[1,3,2] = depth_planes
        cam[1,3,3] = cam[1][3][0] + (cam[1][3][1] * cam[1][3][2])
    elif words_len == 30:
        cam[1,3,0] = float(words[27])
        cam[1,3,1] = float(words[28])
        cam[1,3,2] = float(words[29])
        cam[1,3,3] = cam[1][3][0] + (cam[1][3][1] * cam[1][3][2])
    elif words_len == 31:
        cam[1,3,0] = words[27]
        cam[1,3,1] = float(words[28])
        cam[1,3,2] = float(words[29])
        cam[1,3,3] = float(words[30])
    else:
        cam[1,3,0] = 0
        cam[1,3,1] = 0
        cam[1,3,2] = 0
        cam[1,3,3] = 1

    return cam

def read_cams_sfm(camera_path: str, extension: str = "cam.txt") -> np.ndarray:
    """Reads an entire directory of camera files in SFM format.

    Args:
        camera_path: Path to the directory of camera files.
        extension: File extension being used for the camera files.

    Returns:
        Array of camera extrinsics, intrinsics, and view metadata (Nx2x4x4).
    """
    cam_files = os.listdir(camera_path)
    cam_files.sort()

    cams = []
    
    for cf in cam_files:
        if (cf[-7:] != extension):
            continue

        cam_path = os.path.join(camera_path,cf)
        with open(cam_path,'r') as f:
            cam = read_single_cam_sfm(f, 256)
            cams.append(cam)

    return np.asarray(cams)

def read_cams_trajectory(log_file: str) -> np.ndarray:
    """Reads camera file in Trajectory File format.

    Args:
        log_file: Input *.log file to be read.

    Returns:
        Array of camera extrinsics, intrinsics, and view metadata (Nx2x4x4).
    """
    cam_file = os.path.join(data_path,"camera_pose.log")

    cams = []
    
    with open(cam_file,'r') as f:
        lines = f.readlines()

        for i in range(0,len(lines),5):
            cam = np.zeros((4, 4))
            # read extrinsic
            for j in range(1, 5):
                cam[j-1,:] = np.array([float(l.strip()) for l in lines[i+j].split()])
            cam = np.linalg.inv(cam)
                
            cams.append(cam)

    return cams

def read_extrinsics_tum(tum_file: str, key_frames: List[int] = None) -> np.ndarray:
    """
    """
    rot_interval = 30
    max_rot_angle = math.pi / 3

    extrinsics = []
    with open(tum_file,"r") as tf:
        lines = tf.readlines()
        
        for i,line in enumerate(lines):
            l = np.asarray(line.strip().split(" "), dtype=float)
            l = l[1:]
            t = l[:3]
            q = l[3:]

            R = rot.from_quat(q).as_matrix()
            R = R.transpose()
            t = -R@t
            P = np.zeros((4,4))
            P[:3,:3] = R
            P[:3,3] = t.transpose()
            P[3,3] = 1

            extrinsics.append(P)

            if((key_frames == None) or (i in key_frames)):
                left = np.linspace(0.0, max_rot_angle, rot_interval)
                right = np.linspace(max_rot_angle, -(max_rot_angle), rot_interval*2)
                center = np.linspace(-(max_rot_angle), 0.0, rot_interval)
                thetas = np.concatenate((left,right,center))

                for theta in thetas:
                    new_P = pose_rotation(P,theta)
                    extrinsics.append(new_P)

    return np.asarray(extrinsics)

def read_stereo_intrinsics_yaml(intrinsics_file: str) -> Tuple[ np.ndarray, \
                                                                np.ndarray, \
                                                                np.ndarray, \
                                                                np.ndarray, \
                                                                np.ndarray, \
                                                                np.ndarray]:
    """Reads intrinsics information for a stereo camera pair from a *.yaml file.

    Args:
        intrinsics_file: Input *.yaml file storing the intrinsics information.

    Returns:
        K_left: Intrinsics matrix (3x3) of left camera.
        D_left: Distortion coefficients vector (1x4) of left camera.
        K_right: Intrinsics matrix (3x3) of right camera.
        D_right: Distortion coefficients vector (1x4) of right camera.
        R: Relative rotation matrix (3x3) from left -> right cameras.
        T: Relative translation vector (1x3) from left -> right cameras.
    """
    K_left = np.zeros((3,3))
    D_left = np.zeros((1,4))
    K_right = np.zeros((3,3))
    D_right = np.zeros((1,4))
    R = np.zeros((3,3))
    T = np.zeros((1,3))

    cv_file = cv2.FileStorage(intrinsics_file, cv2.FILE_STORAGE_READ)

    left = cv_file.getNode("left")
    K_left = left.getNode("K").mat()
    D_left = left.getNode("D").mat()

    right = cv_file.getNode("right")
    K_right = right.getNode("K").mat()
    D_right = right.getNode("D").mat()

    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()

    cv_file.release()

    return [K_left, D_left, K_right, D_right, R, t]

def read_matrix(mat_file: str) -> np.ndarray:
    """Reads a single matrix of float values from a file.

    Args:
        mat_file: Input file for the matrix to be read.

    Returns:
        The matrix stored in the given file.
    """
    with open(mat_file, 'r') as f:
        lines = f.readlines()
        M = []

        for l in lines:
            row = l.split()
            row = [float(s) for s in row]
            M.append(row)
        M = np.array(M)

    return M

def write_matrix(M: np.ndarray, mat_file: str) -> None:
    """Writes a single matrix to a file.

    Args:
        M: Matrix to be stored.
        mat_file: Output file where the given matrix is to be writen.
    """
    with open(filename, "w") as f:
        for row in M:
            for e in row:
                f.write("{} ".format(e))
            f.write("\n")

def read_point_cloud(point_cloud_file: str) -> o3d.geometry.PointCloud:
    """Reads a point cloud from a file.

    Args:
        point_cloud_file: Input point cloud file.

    Returns:
        The point cloud stored in the given file.
    """
    return o3d.io.read_point_cloud(point_cloud_file)


def read_mesh(mesh_file: str) -> o3d.geometry.TriangleMesh:
    """Reads a mesh from a file.

    Args:
        mesh_file: Input mesh file.

    Returns:
        The mesh stored in the given file.
    """
    return o3d.io.read_triangle_mesh(mesh_file)

def write_mesh(mesh_file: str, mesh: o3d.geometry.TriangleMesh) -> None:
    """Writes a mesh to a file.

    Args:
        mesh_file: Output mesh file.
        mesh: Mesh to be stored.
    """
    return o3d.io.write_triangle_mesh(mesh_file, mesh)
