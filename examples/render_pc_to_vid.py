import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import sys
import shutil
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as rot


def main():
    if (len(sys.argv) != 7):
        print("Error: usage   python {} <img-width> <img-height> <data-path> <point-cloud-path> <output-path> <video_name>.".format(sys.argv[0]))
        sys.exit()

    width = int(sys.argv[1])
    height = int(sys.argv[2])
    data_path = sys.argv[3]
    pose_file = os.path.join(data_path, "poses.txt")
    intrinsics_file = os.path.join(data_path, "intrinsics.yaml")
    ply_path = sys.argv[4]
    output_path = sys.argv[5]
    vid_name = sys.argv[6]
    img_path = os.path.join(output_path, "images/")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    os.mkdir(img_path)

    # load in mesh
    cloud = read_point_cloud(ply_path)

    # project cloud onto all cameras
    interest_frames = [700,772]
    poses = load_poses(pose_file, interest_frames)
    K = load_intrinsics(intrinsics_file)

    vid_file = os.path.join(output_path, vid_name)
    video = cv2.VideoWriter(vid_file, 0, fps=10, frameSize=((width,height)))

    # set up the renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    render.scene.add_geometry("cloud", cloud, mat)
    render.scene.set_background(np.asarray([255,255,255,1])) #r,g,b,a
    intrins = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])

    with tqdm(poses, unit="pose") as p:
        for view_num, P in enumerate(p):
            # project to depth map
            image = project_cloud(render, intrins, P)

            # store depth map as .pfm
            output_file = os.path.join(img_path, "{:08d}.png".format(view_num))
            cv2.imwrite(output_file, image)

            # write image to video
            video.write(image)

    # clean up video
    cv2.destroyAllWindows()
    video.release()


if __name__=="__main__":
    main()
