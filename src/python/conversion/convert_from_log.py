import numpy as np
from scipy.linalg import null_space
import sys
import cv2
import argparse
import os

# custom imports
sys.path.append("../common_utilities")
from utils import *

# argument parsing
parse = argparse.ArgumentParser(description="Camera Log File Conversion Tool.")

parse.add_argument("-l", "--log_file", default="./data/default_transform.txt", type=str)
parse.add_argument("-o", "--output_path", default="./data/default_transform.txt", type=str)
parse.add_argument("-d", "--old_cam_path", default="./data/default_transform.txt", type=str)

ARGS = parse.parse_args()

def load_cam(cam_file, max_d, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))

    with open(cam_file,'r') as cam_file:
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
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = float(words[27])
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = float(words[29])
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = float(words[29])
        cam[1][3][3] = float(words[30])
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

def write_cam(cam_file, cam):
    f = open(cam_file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

def convert_from_log(log_file, output_path, old_cam_path):
    # write cameras into .log file
    with open(log_file, 'r') as f:
        lines = f.readlines()
        num_lines = len(lines)
        i = 0

        while(i < num_lines-5):
            view_num = int(lines[i].strip().split(" ")[0])
            
            cam = np.zeros((2,4,4))
            cam[0,0,:] = np.asarray(lines[i+1].strip().split(" "), dtype=float)
            cam[0,1,:] = np.asarray(lines[i+2].strip().split(" "), dtype=float)
            cam[0,2,:] = np.asarray(lines[i+3].strip().split(" "), dtype=float)
            cam[0,3,:] = np.asarray(lines[i+4].strip().split(" "), dtype=float)
            cam[0,:,:] = np.linalg.inv(cam[0,:,:])

            cam_file = "{:08d}_cam.txt".format(view_num)
            old_cam  = load_cam(os.path.join(old_cam_path,cam_file), 256)
            new_cam_path = os.path.join(output_path, cam_file)

            cam[1,:,:] = old_cam[1,:,:]

            write_cam(new_cam_path, cam)
            i = i+5
    return

def main():
    # convert cameras into .log file format
    convert_from_log(ARGS.log_file, ARGS.output_path, ARGS.old_cam_path)

if __name__=="__main__":
    main()
