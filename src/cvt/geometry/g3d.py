# geometry/g3d.py

"""Module including routines based in 3D geometry.

This module contains the following functions:

- `render_custom_values(points, values, width, height, cam)` - Renders a point cloud into a 2D camera plane using custom values for each pixel.
- `y_axis_rotation(P, theta)` - Applies a rotation to the given camera extrinsics matrix along the y-axis.
"""

import numpy as np
import os
import sys

import render_points as rp

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

def render_custom_values(points: np.ndarray, values: np.ndarray, width: int, height: int, cam: np.ndarray) -> np.ndarray:
    """Renders a point cloud into a 2D camera plane using custom values for each pixel.

    Parameters:
        points: List of 3D points to be rendered.
        values: List of values to be written in the rendered image.
        width: Desired width of the rendered image.
        height: Desired height of the rendered image.
        cam: Camera parameters for the image viewpoint.

    Returns:
        The rendered image for the list of points using the sepcified corresponding values.
    """
    points = points.tolist()
    values = list(values.astype(float))
    cam = cam.flatten().tolist()

    rendered_img = rp.render(list(image_shape), points, values, cam)

    return rendered_img
