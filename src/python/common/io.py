import os
import sys
import numpy as np
import cv2
import re
import open3d as o3d

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

    return cams

def read_colmap_cams(data_path):
    cam_file = os.path.join(data_path,"camera_poses.log")

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

