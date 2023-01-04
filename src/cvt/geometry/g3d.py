import numpy as np
import os
import sys

import render_points as rp

def pose_rotation(P_old, theta):
    R = np.eye(4)
    R[0,0] = math.cos(theta)
    R[0,2] = math.sin(theta)
    R[2,0] = -(math.sin(theta))
    R[2,2] = math.cos(theta)

    new_P = R @ P_old

    return new_P

def project_cloud(render, intrins, P):
    render.setup_camera(intrins, P)
    image = np.asarray(render.render_to_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def project_2d(points, values, image_shape, cam):
    points = points.tolist()
    values = list(values.astype(float))
    cam = cam.flatten().tolist()

    rendered_img = rp.render(list(image_shape), points, values, cam)

    return rendered_img
