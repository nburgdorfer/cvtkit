"""A suite of common camera utilities.

This module includes several functions for manipulating and extracting information
from camera intrinsics and extrinsics, as well as converting between specific
formats.
"""

import numpy as np
import math
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation
import torch
import open3d as o3d
from typing import Tuple, Any, List
from numpy.typing import NDArray
from torch import Tensor


def build_o3d_traj(
    poses: np.ndarray, K: np.ndarray, width: int, height: int
) -> o3d.camera.PinholeCameraTrajectory:
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
        camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        camera_params.extrinsic = pose
        trajectory.parameters += [camera_params]
    return trajectory


def camera_center(P: NDArray[Any] | Tensor) -> NDArray[Any] | Tensor:
    """Computes the center of a camera in world coordinates.

    Parameters:
        P: The extrinsics matrix (4x4) of a given camera in world-to-camera format.

    Returns:
        The camera center vector (3x1) in world coordinates.
    """
    if isinstance(P, Tensor):
        return _camera_center_torch(P)
    elif isinstance(P, np.ndarray):
        return _camera_center_numpy(P)
    else:
        raise Exception(f"Unknown data type '{type(P)}'")


def _camera_center_numpy(P: NDArray[Any]) -> NDArray[Any]:
    """Numpy version of function 'camera_center'"""
    C = null_space(P[:3, :4])
    C /= C[3, :]

    return C[:3, 0]


def _camera_center_torch(P: Tensor) -> Tensor:
    """PyTorch version of function 'camera_center'"""
    C = tensor_null_space(P)
    C /= C[3, :]

    return C[:3, 0]


def compute_baselines(poses: NDArray[Any] | Tensor) -> Tuple[float, float]:
    """Computes the minimum and maximum baseline between a reference camera pose (poses[0]) and a cluster of supporting camera poses (poses[1:]).

    Parameters:
        poses: Array of camera poses (Nx4x4)

    Returns:
        A tuple of baselines where the first element is the minimum baseline and the second element is the maximum baseline.
    """
    if isinstance(poses, Tensor) or (
        isinstance(poses, List) and isinstance(poses[0], Tensor)
    ):
        return _compute_baselines_torch(poses)
    elif isinstance(poses, np.ndarray) or (
        isinstance(poses, List) and isinstance(poses[0], np.ndarray)
    ):
        return _compute_baselines_numpy(poses)
    else:
        raise Exception(f"Unknown data type '{type(poses)}'")


def _compute_baselines_numpy(
    poses: NDArray[Any] | List[NDArray[Any]],
) -> Tuple[float, float]:
    """Numpy version of function 'compute_baselines'"""
    ref_pose = poses[0]
    ref_camera_center = camera_center(ref_pose[:3, :4])
    min_baseline = float(np.inf)
    max_baseline = float(0.0)

    num_views = len(poses)
    for i in range(1, num_views):
        sup_pose = poses[i]
        sup_camera_center = camera_center(sup_pose[:3, :4])
        b = float(np.linalg.norm(ref_camera_center - sup_camera_center))
        if b < min_baseline:
            min_baseline = b
        if b > max_baseline:
            max_baseline = b
    return (min_baseline, max_baseline)


def _compute_baselines_torch(poses: Tensor | List[Tensor]) -> Tuple[float, float]:
    """PyTorch version of function 'compute_baselines'"""
    ref_pose = poses[0]
    ref_camera_center = camera_center(ref_pose[:3, :4])
    min_baseline = float(torch.inf)
    max_baseline = float(0.0)

    num_views = len(poses)
    for i in range(1, num_views):
        sup_pose = poses[i]
        sup_camera_center = camera_center(sup_pose[:3, :4])
        b = float(torch.linalg.norm(ref_camera_center - sup_camera_center))
        if b < min_baseline:
            min_baseline = b
        if b > max_baseline:
            max_baseline = b
    return (min_baseline, max_baseline)


def crop_intrinsics(
    K: NDArray[Any] | Tensor, crop_row: int, crop_col: int
) -> NDArray[Any] | Tensor:
    """Adjusts intrinsics matrix principle point coordinates corresponding to image cropping.

    Parameters:
        K: 3x3 camera intrinsics matrix.
        crop_row: Offset for cy (corresponding to cropping the left and right side of an image by 'crop_row').
        crop_col: Offset for cx (corresponding to cropping the top and bottom side of an image by 'crop_col').

    Returns:
        The updated 3x3 intrinsics matrix
    """

    K[0, 2] -= crop_col
    K[1, 2] -= crop_row
    return K


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def intrinsic_pyramid(K: NDArray[Any] | Tensor, levels: int) -> NDArray[Any] | Tensor:
    """Computes camera intrinsics pyramid corresonding to several levels of image downsampling.

    Parameters:
        K: [Batch x 3 x 3] camera intrinsics matrix.
        levels: The number of pyramid levels to compute.

    Returns:
        The intrinsics pyramid of shape [Batch x L x 3 x 3] in descending order.
        (L=0 corresponds to the largest image resolution).
    """
    if isinstance(K, Tensor):
        return _intrinsic_pyramid_torch(K, levels)
    elif isinstance(K, np.ndarray):
        return _intrinsic_pyramid_numpy(K, levels)
    else:
        raise Exception(f"Unknown data type '{type(K)}'")


def _intrinsic_pyramid_numpy(K: NDArray[Any], levels: int) -> NDArray[Any]:
    batch_size, _, _ = K.shape
    K_pyr = np.zeros((batch_size, levels, 3, 3)).astype(K.dtype)

    for l in range(levels):
        if l == 0:
            k = K
        else:
            k = np.copy(K) / (2**l)
            k[:, 2, 2] = 1.0
        K_pyr[:, l] = k

    return K_pyr


def _intrinsic_pyramid_torch(K: Tensor, levels: int) -> Tensor:
    batch_size, _, _ = K.shape
    K_pyr = torch.zeros((batch_size, levels, 3, 3)).to(K)

    for l in range(levels):
        if l == 0:
            k = K
        else:
            k = torch.clone(K) / (2**l)
            k[:, 2, 2] = 1.0
        K_pyr[:, l] = k

    return K_pyr


def scale_intrinsics(K: NDArray[Any] | Tensor, scale: float) -> NDArray[Any] | Tensor:
    """Adjust intrinsics matrix focal length and principle point corresponding to image scaling.

    Parameters:
        K: Batchx3x3 camera intrinsics matrices.
        scale: Scale amount (i.e. scale=0.5 corresponds to scaling an image by a factor of 1/2).

    Returns:
        The updated 3x3 intrinsics matrix.
    """
    if isinstance(K, Tensor):
        return _scale_intrinsics_torch(K, scale)
    elif isinstance(K, np.ndarray):
        return _scale_intrinsics_numpy(K, scale)
    else:
        raise Exception(f"Unknown data type '{type(K)}'")


def _scale_intrinsics_torch(K: Tensor, scale: float) -> Tensor:
    K_clone = torch.clone(K)
    K_clone[:, :2, :] = K_clone[:, :2, :] * scale

    return K_clone


def _scale_intrinsics_numpy(K: NDArray[Any], scale: float) -> NDArray[Any]:
    K_copy = K.copy()
    K_copy[:2, :] = K_copy[:2, :] * scale

    return K_copy


def tensor_null_space(A: Tensor) -> Tensor:
    """Computes the null space of a PyTorch tensor.

    Parameters:
        A: PyTorch tensor of shape m x n.

    Returns:
        The null space of tensor A.
    """
    _, S, V = torch.linalg.svd(A, full_matrices=True)
    tol = torch.max(S) * torch.finfo(S.dtype).eps * max(A.shape)
    rank = torch.sum(S > tol)

    return V[rank:].T.conj()


def to_opengl_pose(pose: NDArray[Any] | Tensor) -> NDArray[Any] | Tensor:
    """OpenGL pose: (right-up-back) (cam-to-world)

    Parameters:
        pose: The pose in non-OpenGL format

    Returns:
        Camera pose in OpenGL format.
    """
    if isinstance(pose, Tensor):
        return _to_opengl_pose_torch(pose)
    elif isinstance(pose, np.ndarray):
        return _to_opengl_pose_numpy(pose)
    else:
        raise Exception(f"Unknown data type '{type(pose)}'")


def _to_opengl_pose_torch(pose: Tensor) -> Tensor:
    pose = torch.linalg.inv(pose)
    pose[:3, 1] *= -1
    pose[:3, 2] *= -1
    return pose


def _to_opengl_pose_numpy(pose: NDArray[Any]) -> NDArray[Any]:
    pose = np.linalg.inv(pose)
    pose[:3, 1] *= -1
    pose[:3, 2] *= -1
    return pose
