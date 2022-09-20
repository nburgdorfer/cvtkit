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

def read_point_cloud(ply_path, size=0.1):
    if(ply_path[-3:] != "ply"):
        print("{} is not a '.ply' file.".format(ply_path))
        sys.exit()

    ply = o3d.io.read_point_cloud(ply_path)
    #ply = ply.voxel_down_sample(voxel_size=size)

    return ply

def load_poses(pose_file):
    poses = []
    with open(pose_file,"r") as pf:
        lines = pf.readlines()
        
        for line in lines:
            l = np.asarray(line.strip().split(" "), dtype=float)
            l = l[1:]
            t = l[:3]
            q = l[3:]

            R = rot.from_quat(q).as_matrix()
            R = R.transpose()
            t = -R@t
            P = np.zeros((4,4))
            P[:3,:3] = R
            P[:3,3] = t.transpose()
            P[3,3] = 1

            poses.append(P)

    return np.asarray(poses)

def load_intrinsics(intrinsics_file):
    K = np.eye(3)
    cv_file = cv2.FileStorage(intrinsics_file, cv2.FILE_STORAGE_READ)
    left = cv_file.getNode("left")
    K = left.getNode("K").mat()
    cv_file.release()

    return K

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

def project_cloud(render, intrins, P):
    render.setup_camera(intrins, P)
    image = np.asarray(render.render_to_image())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

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
    poses = load_poses(pose_file)
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
