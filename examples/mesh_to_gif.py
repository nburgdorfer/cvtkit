import sys
import os
import argparse
from tqdm import tqdm

import cv2
import numpy as np
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
        help="The name of the output video file (with extension). This will be stored in the output path (ex: mesh_vid.gif)")
parse.add_argument("-w", "--width", default=1600, type=int,
        help="The desired image width.")
parse.add_argument("-t", "--height", default=1200, type=int,
        help="The desired image height.")
parse.add_argument("-f", "--fps", default=8, type=int,
        help="The desired frame rate for the captured video.")
parse.add_argument("-s", "--offset_index", default=0, type=int,
        help="The desired frame index to begin the gif animation.")
parse.add_argument("-n", "--total_frames", default=8, type=int,
        help="The desired number of frames to include in the animation (loops forward and backward through the frames).")
ARGS = parse.parse_args()


def main():
    if not os.path.exists(ARGS.output_path):
        os.mkdir(ARGS.output_path)

    # load in data
    mesh = read_mesh(ARGS.mesh)
    cams = read_cams_sfm(ARGS.camera_path)

    # set up the renderer
    render = o3d.visualization.rendering.OffscreenRenderer(ARGS.width, ARGS.height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    render.scene.add_geometry("mesh", mesh, mat)
    render.scene.set_background(np.asarray([0,0,0,1])) #r,g,b,a
    video_file = os.path.join(ARGS.output_path, ARGS.video_file)

    # compute list of indices
    indices = list(range(0,ARGS.total_frames,1)) + list(range(ARGS.total_frames,0,-1))

    # animate scene
    with imageio.get_writer(video_file, mode="I", fps=ARGS.fps) as writer:
        for view_num in tqdm(indices):
            # get camera for current view
            cam = cams[ARGS.offset_index + view_num]

            # project to image
            image = project_renderer(render, cam[1], cam[0], ARGS.width, ARGS.height)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # write image to gif
            writer.append_data(image)

if __name__=="__main__":
    main()
