import sys, os
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
from numpy import inf

from cvt.geometry import render_point_cloud
from cvt.io import read_cams_sfm, read_point_cloud, write_pfm


def main():
    point_cloud_file = sys.argv[1]
    cam_path = sys.argv[2]
    image_path = sys.argv[3]
    output_path = sys.argv[4]

    # get image shape
    image_files = os.listdir(image_path)
    image_files.sort()
    img = cv2.imread(os.path.join(image_path,image_files[0]))
    h,w,_ = img.shape

    # read in data
    cams = read_cams_sfm(cam_path)

    # read point cloud
    cloud = read_point_cloud(point_cloud_file)

    # set up the renderer
    render = o3d.visualization.rendering.OffscreenRenderer(w, h)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    render.scene.add_geometry("cloud", cloud, mat)
    render.scene.set_background(np.asarray([0,0,0,1])) #r,g,b,a
    
    for i,cam in enumerate(cams):
        intrins = o3d.camera.PinholeCameraIntrinsic(w, h, cam[1,0,0], cam[1,1,1], cam[1,0,2], cam[1,1,2])
        _, depth = render_point_cloud(render, intrins, cam[0])
        depth = np.nan_to_num(depth)
        depth[depth>=1e5] = 0.0
        write_pfm(os.path.join(output_path,f"{i:08d}.pfm"), depth)

if __name__=="__main__":
    main()
