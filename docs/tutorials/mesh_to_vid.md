This script creates a video of a mesh following the provided camera trajectories.

## Code
```python
import sys
import os
from tqdm import tqdm
import argparse

import numpy as np
import cv2
import imageio

import open3d as o3d
import open3d.visualization.rendering as rendering

from cvt.io import read_mesh, read_cams_sfm
from cvt.geometry import project_renderer

# argument parsing
parse = argparse.ArgumentParser(description="Mesh-Video generator.")
parse.add_argument("-m", "--mesh", default="mesh.ply", type=str,
        help="Input mesh to be captured in video.")
parse.add_argument("-o", "--output_path", default="images/", type=str,
        help="Output directory where all frames of the video will be stored.")
parse.add_argument("-c", "--camera_path", default="cameras/", type=str,
        help="Input path to the directory where the camera information is stored.")
parse.add_argument("-v", "--video_file", default="", type=str,
        help="The name of the output video file (with extension). This will be stored in the output path (ex: mesh_vid.avi)")
parse.add_argument("-w", "--width", default=1600, type=int,
        help="The desired image width.")
parse.add_argument("-t", "--height", default=1200, type=int,
        help="The desired image height.")
parse.add_argument("-f", "--fps", default=10, type=int,
        help="The desired frame rate for the captured video.")
ARGS = parse.parse_args()


def main():
    if not os.path.exists(ARGS.output_path):
        os.mkdir(ARGS.output_path)

    # load in data
    mesh = read_mesh(ARGS.mesh)
    cams = read_cams_sfm(ARGS.camera_path)

    # create video writer
    if (ARGS.video_file[-3:] != "avi"):
        print("Error: expected a *.avi filename.")
        sys.exit()
    video_file = os.path.join(ARGS.output_path, ARGS.video_file)
    video = cv2.VideoWriter(video_file, 0, fps=ARGS.fps, frameSize=((ARGS.width,ARGS.height)))

    # set up the renderer
    render = o3d.visualization.rendering.OffscreenRenderer(ARGS.width, ARGS.height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    render.scene.add_geometry("mesh", mesh, mat)
    render.scene.set_background(np.asarray([0,0,0,1])) #r,g,b,a

    with tqdm(cams, unit="view") as cam_iter:
        for view_num, cam in enumerate(cam_iter):
            # project to image
            image = project_renderer(render, cam[1], cam[0], ARGS.width, ARGS.height)

            # write image to video
            video.write(image)

    # clean up video
    cv2.destroyAllWindows()
    video.release()

if __name__=="__main__":
    main()
```
