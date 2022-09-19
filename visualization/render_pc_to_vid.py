import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import sys
import shutil
import os

def load_cam(cam_file):
    K = np.zeros((3,3))
    P = np.zeros((4,4))

    words = cam_file.read().split()

    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            P[i,j] = float(words[extrinsic_index])

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            K[i,j] = float(words[intrinsic_index])

    return K, P

def write_pfm(pfm_file, image, scale=1):
    pfm_file = open(pfm_file, 'wb')
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

    image_string = image.tobytes()
    pfm_file.write(image_string)

    pfm_file.close()

def project_mesh(mesh, K, P, width, height):
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'

    render.scene.add_geometry("mesh", mesh, mat)
    intrins = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
    render.setup_camera(intrins, P)

    depth_map = np.asarray(render.render_to_depth_image(z_in_view_space=True))
    depth_map = np.where(np.isinf(depth_map), 0, depth_map)

    return depth_map

def main():
    if (len(sys.argv) != 6):
        print("Error: usage   python {} <img-width> <img-height> <camera-dir> <mesh-path> <output-path>.".format(sys.argv[0]))
        sys.exit()


    width = int(sys.argv[1])
    height = int(sys.argv[2])
    cam_path = sys.argv[3]
    mesh_path = sys.argv[4]
    output_path = sys.argv[5]
    disp_path = os.path.join(output_path, "disp/")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    os.mkdir(disp_path)

    # load in mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # project mesh onto all cameras
    cam_files = os.listdir(cam_path)
    cam_files.sort()

    for cf in cam_files:
        if (cf[-7:] == "cam.txt"):
            # load camera
            cam_file = os.path.join(cam_path, cf)
            K, P = load_cam(open(cam_file,'r')) 

            # project to depth map
            depth_map = project_mesh(mesh, K, P, width, height)

            # store depth map as .pfm
            view_num = int(cf[:8])
            output_file = os.path.join(output_path, "{:08d}_depth.pfm".format(view_num))
            write_pfm(output_file, depth_map)

            m = np.max(depth_map)+1e-7
            disp = os.path.join(disp_path, "{:08d}_depth_disp.png".format(view_num))
            cv2.imwrite(disp, depth_map/m*255)

if __name__=="__main__":
    main()
