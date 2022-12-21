"""
Computes the transformation between two camera systems.
It includes the option to incorporate an additional transformation
which will be left-multiplied to the resultant transformation between
system 1 and system 2. For example, if you would like to compute the
transformation between a camera system A and a camera system B, do not
include an alignment file; the resulting matrix produced will be the
transformation A -> B. If you would like to include an addition 
transformation, lets say B -> W, then include alignment file for B -> W;
the resulting matrix will be the transformation A -> B -> W.

This script also produces an identity transformation in the same
directory as specified in the '--output_file' option.
This file will be named 'identity_trans.txt' and is produced
for convienence if needed; (for example, if no transformation is
needed, but an alignment file must be passed to some other script).
"""

import sys
import os
import numpy as np
import cv2
import argparse
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# custom imports
from cvt.common.io import *

# argument parsing
parse = argparse.ArgumentParser(description="Camera System Alignment Tool.")

parse.add_argument("-a", "--data_path_1", default="/data/dtu/a/cams", type=str, help="Path to the camera data for the first format.")
parse.add_argument("-b", "--data_path_2", default="/data/dtu/b/cams", type=str, help="Path to the camera data for the second format.")
parse.add_argument("-f", "--format_1", default="mvsnet", type=str, help="The format type for the first stored camera data (e.g. mvsnet, colmap, ...).")
parse.add_argument("-g", "--format_2", default="colmap", type=str, help="The format type for the second stored camera data (e.g. mvsnet, colmap, ...).")
parse.add_argument("-t", "--alignment", default=None, type=str, help="The path to an additional alignment from format 2 to some other coordinate system (e.g. colmap -> ground-truth).")
parse.add_argument("-o", "--output_file", default="./data/default_transform.txt", type=str, help="The path to the output file to store the transformation.")

ARGS = parse.parse_args()

def compute_alignment(cams_1, cams_2, A):
    centers_1 = np.squeeze(np.array([ camera_center(c) for c in cams_1 ]), axis=2)
    centers_2 = np.squeeze(np.array([ camera_center(c) for c in cams_2 ]), axis=2)

    ### determine scale
    # grab first camera pair
    c1_0 = centers_1[0][:3]
    c2_0 = centers_2[0][:3]

    # grab one-hundreth camera pair
    c1_1 = centers_1[99][:3]
    c2_1 = centers_2[99][:3]

    # calculate the baseline between both sets of cameras
    baseline_1 = np.linalg.norm(c1_0 - c1_1)
    baseline_2 = np.linalg.norm(c2_0 - c2_1)

    # compute the scale based on the baseline ratio
    scale = baseline_2/baseline_1
    print("Scale: ", scale)

    ### determine 1->2 Rotation 
    b1 = np.array([c[:3] for c in centers_1])
    b2 = np.array([c[:3] for c in centers_2])
    R = Rotation.align_vectors(b2,b1)[0].as_matrix()
    R = scale * R

    ### create transformation matrix
    M = np.eye(4)
    M[:3,:3] = R

    ### determine 1->2 Translation
    num_cams = len(cams_1)
    t = np.array([ c2-(M@c1) for c1,c2 in zip(centers_1,centers_2) ])
    t = np.mean(t, axis=0)

    ### add translation
    M[:3,3] = t[:3]

    ### apply additional alignment
    M = A@M
    print("Resulting Transformation:\n", M)

    return M

def main():
    # read in format 1 cameras
    if (ARGS.format_1 == "mvsnet"):
        cams_1 = load_mvsnet_cams(ARGS.data_path_1)
    elif(ARGS.format_1 == "colmap"):
        cams_1 = load_colmap_cams(ARGS.data_path_1)
    else:
        print("ERROR: unknown format type '{}'".format(form))
        sys.exit()

    # read in format 2 cameras
    if (ARGS.format_2 == "mvsnet"):
        cams_2 = load_mvsnet_cams(ARGS.data_path_2)
    elif(ARGS.format_2 == "colmap"):
        cams_2 = load_colmap_cams(ARGS.data_path_2)
    else:
        print("ERROR: unknown format type '{}'".format(form))
        sys.exit()

    # load additional alignment file
    if (ARGS.alignment == None):
        A = np.eye(4)
    else:
        A = read_matrix(ARGS.alignment)

    # compute the alignment between the two systems
    M = compute_relative_transform(cams_1, cams_2, A)
    write_matrix(M, ARGS.output_file)


    dir_ind = ARGS.output_file.rfind("/")
    I_output = ARGS.output_file[:(dir_ind+1)]+"identity_trans.txt"

    write_matrix(np.eye(4), I_output)
        

if __name__=="__main__":
    main()
