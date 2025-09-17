# cvt/io.py

"""A suite of common input/output functions.

This module includes several functions for reading and writing
different types of data useful for computer vision applications.
"""

import os
import sys
import numpy as np
import cv2
import re
import math
import open3d as o3d
import torch
from typing import List, Tuple, Any
from numpy.typing import NDArray
from torch import Tensor

from scipy.spatial.transform import Rotation as rot

from cvtkit.common import y_axis_rotation
from cvtkit.camera import fov2focal


def read_cams_sfm(camera_path: str, extension: str = "cam.txt") -> np.ndarray:
    """Reads an entire directory of camera files in SFM format.

    Parameters:
        camera_path: Path to the directory of camera files.
        extension: File extension being used for the camera files.

    Returns:
        Array of camera extrinsics, intrinsics, and view metadata (Nx2x4x4).
    """
    cam_files = os.listdir(camera_path)
    cam_files.sort()

    cams = []

    for cf in cam_files:
        if cf[-7:] != extension:
            continue

        cam_path = os.path.join(camera_path, cf)
        # with open(cam_path,'r') as f:
        cam = read_single_cam_sfm(cam_path, 256)
        cams.append(cam)

    return np.asarray(cams)


def read_cams_trajectory(log_file: str) -> NDArray[np.float32]:
    """Reads camera file in Trajectory File format.

    Parameters:
        log_file: Input *.log file to be read.

    Returns:
        Array of camera extrinsics, intrinsics, and view metadata (Nx2x4x4).
    """
    cams = []

    with open(log_file, "r") as f:
        lines = f.readlines()

        for i in range(0, len(lines), 5):
            cam = np.zeros((4, 4))
            # read extrinsic
            for j in range(1, 5):
                cam[j - 1, :] = np.array(
                    [float(l.strip()) for l in lines[i + j].split()]
                )
            cam = np.linalg.inv(cam)

            cams.append(cam)

    return np.asarray(cams, dtype=np.float32)


def read_extrinsics_tum(
    tum_file: str, key_frames: List[int] | None = None
) -> np.ndarray:
    """Reads extrinsic camera trajectories in TUM format [timestamp tx ty tz qx qy qz qw].

    Parameters:
        tum_file: Input extrinsics file.
        key_frames: Indices corresponding to the desired keyframes.

    Returns:
        Array of camera extrinsics (Nx4x4).
    """
    rot_interval = 30
    max_rot_angle = math.pi / 3

    extrinsics = []
    with open(tum_file, "r") as tf:
        lines = tf.readlines()

        for i, line in enumerate(lines):
            l = np.asarray(line.strip().split(" "), dtype=float)
            l = l[1:]
            t = l[:3]
            q = l[3:]

            R = rot.from_quat(q).as_matrix()
            R = R.transpose()
            t = -R @ t
            P = np.zeros((4, 4))
            P[:3, :3] = R
            P[:3, 3] = t.transpose()
            P[3, 3] = 1

            extrinsics.append(P)

            if (key_frames == None) or (i in key_frames):
                left = np.linspace(0.0, max_rot_angle, rot_interval)
                right = np.linspace(max_rot_angle, -(max_rot_angle), rot_interval * 2)
                center = np.linspace(-(max_rot_angle), 0.0, rot_interval)
                thetas = np.concatenate((left, right, center))

                for theta in thetas:
                    new_P = y_axis_rotation(P, theta)
                    extrinsics.append(new_P)

    return np.asarray(extrinsics)


def read_matrix(mat_file: str) -> np.ndarray:
    """Reads a single matrix of float values from a file.

    Parameters:
        mat_file: Input file for the matrix to be read.

    Returns:
        The matrix stored in the given file.
    """
    with open(mat_file, "r") as f:
        lines = f.readlines()
        M = []

        for l in lines:
            row = l.split()
            row = [float(s) for s in row]
            M.append(row)
        M = np.array(M)

    return M


def read_mesh(mesh_file: str) -> o3d.geometry.TriangleMesh:
    """Reads a mesh from a file.

    Parameters:
        mesh_file: Input mesh file.

    Returns:
        The mesh stored in the given file.
    """
    return o3d.io.read_triangle_mesh(mesh_file)


def read_cluster_list(filename: str) -> List[Tuple[int, List[int]]]:
    """Reads a cluster list file encoding supporting camera viewpoints.

    Parameters:
        filename: Input file encoding per-camera viewpoints.

    Returns:
        An array of tuples encoding (ref_view, [src_1,src_2,..])
    """
    data = []
    with open(filename) as f:
        num_views = int(f.readline())
        all_views = list(range(0, num_views))

        for view_idx in range(num_views):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) == 0:
                continue
            data.append((ref_view, src_views))
    return data


def read_pfm(pfm_file: str) -> NDArray[np.float32]:
    """Reads a file in *.pfm format.

    Parameters:
        pfm_file: Input *.pfm file to be read.

    Returns:
        Data map that was stored in the *.pfm file.
    """
    with open(pfm_file, "rb") as f:
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = f.readline().decode("iso8859_15").rstrip()

        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("iso8859_15"))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")
        # scale = float(file.readline().rstrip())
        scale = float((f.readline()).decode("iso8859_15").rstrip())
        if scale < 0:  # little-endian
            data_type = "<f"
        else:
            data_type = ">f"  # big-endian
        data_string = f.read()
        data = np.frombuffer(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
    return data.astype(np.float32)


def read_point_cloud(point_cloud_file: str) -> o3d.geometry.PointCloud:
    """Reads a point cloud from a file.

    Parameters:
        point_cloud_file: Input point cloud file.

    Returns:
        The point cloud stored in the given file.
    """
    return o3d.io.read_point_cloud(point_cloud_file)


def read_point_cloud_np(point_cloud_file: str) -> NDArray[np.float32]:
    """Reads a point cloud from a file.

    Parameters:
        point_cloud_file: Input point cloud file.

    Returns:
        The point cloud as an NDArray stored in the given file.
    """
    cloud = o3d.io.read_point_cloud(point_cloud_file)
    return np.array(cloud.points, dtype=np.float32)


def read_single_cam_sfm(cam_file: str, depth_planes: int = 256) -> np.ndarray:
    """Reads a single camera file in SFM format.

    Parameters:
        cam_file: Input camera file to be read.
        depth_planes: Number of depth planes to store in the view metadata.

    Returns:
        Camera extrinsics, intrinsics, and view metadata (2x4x4).
    """
    cam = np.zeros((2, 4, 4))

    with open(cam_file, "r") as f:
        words = f.read().split()

    words_len = len(words)

    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0, i, j] = float(words[extrinsic_index])

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1, i, j] = float(words[intrinsic_index])

    if words_len == 29:
        cam[1, 3, 0] = float(words[27])
        cam[1, 3, 1] = float(words[28])
        cam[1, 3, 2] = depth_planes
        cam[1, 3, 3] = cam[1][3][0] + (cam[1][3][1] * cam[1][3][2])
    elif words_len == 30:
        cam[1, 3, 0] = float(words[27])
        cam[1, 3, 1] = float(words[28])
        cam[1, 3, 2] = float(words[29])
        cam[1, 3, 3] = cam[1][3][0] + (cam[1][3][1] * cam[1][3][2])
    elif words_len == 31:
        cam[1, 3, 0] = words[27]
        cam[1, 3, 1] = float(words[28])
        cam[1, 3, 2] = float(words[29])
        cam[1, 3, 3] = float(words[30])
    else:
        cam[1, 3, 0] = 0
        cam[1, 3, 1] = 0
        cam[1, 3, 2] = 0
        cam[1, 3, 3] = 1

    return cam


def read_stereo_intrinsics_yaml(
    intrinsics_file: str,
) -> Tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """Reads intrinsics information for a stereo camera pair from a *.yaml file.

    Parameters:
        intrinsics_file: Input *.yaml file storing the intrinsics information.

    Returns:
        K_left: Intrinsics matrix (3x3) of left camera.
        D_left: Distortion coefficients vector (1x4) of left camera.
        K_right: Intrinsics matrix (3x3) of right camera.
        D_right: Distortion coefficients vector (1x4) of right camera.
        R: Relative rotation matrix (3x3) from left -> right cameras.
        T: Relative translation vector (1x3) from left -> right cameras.
    """
    K_left = np.zeros((3, 3))
    D_left = np.zeros((1, 4))
    K_right = np.zeros((3, 3))
    D_right = np.zeros((1, 4))
    R = np.zeros((3, 3))
    T = np.zeros((1, 3))

    cv_file = cv2.FileStorage(intrinsics_file, cv2.FILE_STORAGE_READ)

    left = cv_file.getNode("left")
    K_left = np.asarray(left.getNode("K").mat(), dtype=np.float32)
    D_left = np.asarray(left.getNode("D").mat(), dtype=np.float32)

    right = cv_file.getNode("right")
    K_right = np.asarray(right.getNode("K").mat(), dtype=np.float32)
    D_right = np.asarray(right.getNode("D").mat(), dtype=np.float32)

    R = np.asarray(cv_file.getNode("R").mat(), dtype=np.float32)
    T = np.asarray(cv_file.getNode("T").mat(), dtype=np.float32)

    cv_file.release()

    return (K_left, D_left, K_right, D_right, R, T)


def write_cam_sfm(
    cam_file: str, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> None:
    """Writes intrinsic and extrinsic camera parameters to a file in sfm format.

    Parameters:
        cam_file: The file to be writen to.
        intrinsics: Camera intrinsic data to be written.
        extrinsics: Camera extrinsic data to be written.
    """
    with open(cam_file, "w") as f:
        f.write("extrinsic\n")
        for i in range(0, 4):
            for j in range(0, 4):
                f.write(str(extrinsics[i][j]) + " ")
            f.write("\n")
        f.write("\n")

        f.write("intrinsic\n")
        for i in range(0, 3):
            for j in range(0, 3):
                f.write(str(intrinsics[i][j]) + " ")
            f.write("\n")


def write_matrix(M: np.ndarray, mat_file: str) -> None:
    """Writes a single matrix to a file.

    Parameters:
        M: Matrix to be stored.
        mat_file: Output file where the given matrix is to be writen.
    """
    with open(mat_file, "w") as f:
        for row in M:
            for e in row:
                f.write("{} ".format(e))
            f.write("\n")


def write_mesh(mesh_file: str, mesh: o3d.geometry.TriangleMesh) -> None:
    """Writes a mesh to a file.

    Parameters:
        mesh_file: Output mesh file.
        mesh: Mesh to be stored.
    """
    return o3d.io.write_triangle_mesh(mesh_file, mesh)


def write_pfm(filename: str, data_map: np.ndarray, scale: float = 1.0) -> None:
    """Writes a data map to a file in *.pfm format.

    Parameters:
        pfm_file: Output *.pfm file to store the data map.
        data_map: Data map to be stored.
        scale: Value used to scale the data map.
    """
    with open(filename, "wb") as pfm_file:
        color = None

        if data_map.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        data_map = np.flipud(data_map)

        if len(data_map.shape) == 3 and data_map.shape[2] == 3:  # color data_map
            color = True
        elif len(data_map.shape) == 2 or (
            len(data_map.shape) == 3 and data_map.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        a = "PF\n" if color else "Pf\n"
        b = "%d %d\n" % (data_map.shape[1], data_map.shape[0])

        pfm_file.write(a.encode("iso8859-15"))
        pfm_file.write(b.encode("iso8859-15"))

        endian = data_map.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        c = "%f\n" % scale
        pfm_file.write(c.encode("iso8859-15"))

        data_map_string = data_map.tobytes()
        pfm_file.write(data_map_string)


def write_point_cloud(fn, cloud):
    o3d.io.write_point_cloud(fn, cloud)


def write_point_cloud_np(filename, points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, cloud)


def save_state_dict(model, save_path):
    torch.save(model.state_dict(), save_path)


def load(model, load_path):
    model.load_state_dict(torch.load(load_path))


def save_ckpt(model, save_path):
    save_dict = {"model": model.state_dict()}
    torch.save(save_dict, save_path)


def load_ckpt(model, load_path, strict=True):
    model_dict = torch.load(load_path)
    model.load_state_dict(model_dict["model"], strict=strict)


def load_pretrained_model(model, ckpt):
    """Loads model weights from disk."""
    print(f"Loading model from: {ckpt}...")
    try:
        model.load_state_dict(torch.load(ckpt))
    except Exception as e:
        print(e)
        print("Failed loading network weights...")
        sys.exit()


def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera["R"].transpose()
    Rt[:3, 3] = camera["T"]
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": f"{id:08d}",
        "width": camera["width"],
        "height": camera["height"],
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera["FovY"], camera["height"]),
        "fx": fov2focal(camera["FovX"], camera["width"]),
    }
    return camera_entry


def trajectory_to_sfm(log_file: str, camera_path: str, intrinsics: np.ndarray) -> None:
    """Convert a set of cameras from Trajectory File format to SFM format.

    Args:
        log_file: Input *.log file that stores the trajectory information.
        camera_path: Output path where the SFM camera files will be written.
        intrinsics: Array of intrinsics matrices (Nx3x3) for each camera.
    """
    with open(log_file, "r") as f:
        lines = f.readlines()
        num_lines = len(lines)
        i = 0

        while i < num_lines - 5:
            view_num = int(lines[i].strip().split(" ")[0])

            cam = np.zeros((2, 4, 4))
            cam[0, 0, :] = np.asarray(lines[i + 1].strip().split(" "), dtype=float)
            cam[0, 1, :] = np.asarray(lines[i + 2].strip().split(" "), dtype=float)
            cam[0, 2, :] = np.asarray(lines[i + 3].strip().split(" "), dtype=float)
            cam[0, 3, :] = np.asarray(lines[i + 4].strip().split(" "), dtype=float)
            cam[0, :, :] = np.linalg.inv(cam[0, :, :])

            cam_file = "{:08d}_cam.txt".format(view_num)
            cam_path = os.path.join(camera_path, cam_file)

            cam[1, :, :] = intrinsics[view_num]

            write_cam_sfm(cam_path, cam[0], cam[1])
            i = i + 5
    return


def sfm_to_trajectory(cams: NDArray[Any] | Tensor, log_file: str) -> None:
    """Convert a set of cameras from SFM format to Trajectory File format.

    Parameters:
        cams: Array of camera extrinsics (Nx4x4) to be converted.
        log_file: Output path to the *.log file that is to be created.
    """
    num_cams = len(cams)

    with open(log_file, "w") as f:
        for i, cam in enumerate(cams):
            # write camera to output_file
            f.write("{} {} 0\n".format(str(i), str(i)))
            for row in cam:
                for c in row:
                    f.write("{} ".format(str(c)))
                f.write("\n")

    return
