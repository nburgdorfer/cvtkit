import os
import sys
import numpy as np
import cv2
import re
import open3d as o3d

#### TEMP - NSF underwater things ####
def load_poses(pose_file, interest_frames):
    rot_interval = 30
    max_rot_angle = math.pi / 3

    poses = []
    with open(pose_file,"r") as pf:
        lines = pf.readlines()
        
        for i,line in enumerate(lines):
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

            if(i in interest_frames):
                left = np.linspace(0.0, max_rot_angle, rot_interval)
                right = np.linspace(max_rot_angle, -(max_rot_angle), rot_interval*2)
                center = np.linspace(-(max_rot_angle), 0.0, rot_interval)
                thetas = np.concatenate((left,right,center))

                for theta in thetas:
                    new_P = pose_rotation(P,theta)
                    poses.append(new_P)

    return np.asarray(poses)
def load_intrinsics(intrinsics_file):
    K = np.eye(3)
    cv_file = cv2.FileStorage(intrinsics_file, cv2.FILE_STORAGE_READ)
    left = cv_file.getNode("left")
    K = left.getNode("K").mat()
    cv_file.release()

    return K
#### TEMP - NSF underwater things ####





def read_pfm(pfm_file):
    with open(pfm_file, 'rb') as pfm_file:
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = pfm_file.readline().decode('iso8859_15').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('iso8859_15'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        # scale = float(file.readline().rstrip())
        scale = float((pfm_file.readline()).decode('iso8859_15').rstrip())
        if scale < 0: # little-endian
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = pfm_file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
    return data

def read_cam(cam_file, max_d=256, interval_scale=1):
    cam = np.zeros((2, 4, 4))
    words = cam_file.read().split()

    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = float(words[extrinsic_index])

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = float(words[intrinsic_index])

    if len(words) == 29:
        cam[1][3][0] = float(words[27])
        cam[1][3][1] = float(words[28])
        cam[1][3][2] = max_d
        cam[1][3][3] = cam[1][3][0] + (cam[1][3][1] * cam[1][3][2])
    elif len(words) == 30:
        cam[1][3][0] = float(words[27])
        cam[1][3][1] = float(words[28])
        cam[1][3][2] = float(words[29])
        cam[1][3][3] = cam[1][3][0] + (cam[1][3][1] * cam[1][3][2])
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28])
        cam[1][3][2] = float(words[29])
        cam[1][3][3] = float(words[30])
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

def read_cams(data_path, extension="cam.txt"):
    cam_files = os.listdir(data_path)
    cam_files.sort()

    cams = []
    
    for cf in cam_files:
        if (cf[-7:] != extension):
            continue

        cam_path = os.path.join(data_path,cf)
        with open(cam_path,'r') as f:
            cam = load_cam(f, 256)
            cams.append(cam)

    return cams

def read_mvsnet_cams(data_path):
    cam_files = os.listdir(data_path)
    cam_files.sort()

    cams = []
    
    for cf in cam_files:
        if (cf[-7:] != "cam.txt"):
            continue

        cam_path = os.path.join(data_path,cf)
        with open(cam_path,'r') as f:
            cam = np.zeros((4, 4))
            words = f.read().split()

            # read extrinsic
            for i in range(0, 4):
                for j in range(0, 4):
                    extrinsic_index = 4 * i + j + 1
                    cam[i][j] = float(words[extrinsic_index])
                    
            cams.append(cam)

    return np.asarray(cams)

def read_colmap_cams(data_path):
    cam_file = os.path.join(data_path,"camera_pose.log")

    cams = []
    
    with open(cam_file,'r') as f:
        lines = f.readlines()

        for i in range(0,len(lines),5):
            cam = np.zeros((4, 4))
            # read extrinsic
            for j in range(1, 5):
                cam[j-1,:] = np.array([float(l.strip()) for l in lines[i+j].split()])
            cam = np.linalg.inv(cam)
                
            cams.append(cam)

    return cams

def read_matrix(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        M = []

        for l in lines:
            row = l.split()
            row = [float(s) for s in row]
            M.append(row)
        M = np.array(M)

    return M

def write_matrix(M, filename):
    with open(filename, "w") as f:
        for row in M:
            for e in row:
                f.write("{} ".format(e))
            f.write("\n")

def read_point_cloud(ply_path):
    if(ply_path[-3:] != "ply"):
        print("Error: file {} is not a '.ply' file.".format(ply_path))

    return o3d.io.read_point_cloud(ply_path, format="ply")


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

    image_string = image.tostring()
    pfm_file.write(image_string)

    pfm_file.close()
