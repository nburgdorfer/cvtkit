import numpy as np
import os
import sys

import render_points as rp

def project_2d(points, values, image_shape, cam):
    points = points.tolist()
    values = list(values.astype(float))
    cam = cam.flatten().tolist()

    rendered_img = rp.render(list(image_shape), points, values, cam)

    return rendered_img
