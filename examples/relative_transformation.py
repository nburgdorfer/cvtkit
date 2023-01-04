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

# custom imports
from cvt.common.io import *
from cvt.common.camera import *

# argument parsing
parse = argparse.ArgumentParser(description="Camera System Alignment Tool.")

parse.add_argument("-a", "--data_path_1", default="/data/dtu/a/cams", type=str, help="Path to the camera data for the first format.")
parse.add_argument("-b", "--data_path_2", default="/data/dtu/b/cams", type=str, help="Path to the camera data for the second format.")
parse.add_argument("-f", "--format_1", default="mvsnet", type=str, help="The format type for the first stored camera data (e.g. mvsnet, colmap, ...).")
parse.add_argument("-g", "--format_2", default="colmap", type=str, help="The format type for the second stored camera data (e.g. mvsnet, colmap, ...).")
parse.add_argument("-t", "--alignment", default=None, type=str, help="The path to an additional alignment from format 2 to some other coordinate system (e.g. colmap -> ground-truth).")
parse.add_argument("-o", "--output_file", default="./data/default_transform.txt", type=str, help="The path to the output file to store the transformation.")

ARGS = parse.parse_args()

def main():
    # read in format 1 cameras
    if (ARGS.format_1 == "mvsnet"):
        cams_1 = read_mvsnet_cams(ARGS.data_path_1)
    elif(ARGS.format_1 == "colmap"):
        cams_1 = read_colmap_cams(ARGS.data_path_1)
    else:
        print("ERROR: unknown format type '{}'".format(form))
        sys.exit()

    # read in format 2 cameras
    if (ARGS.format_2 == "mvsnet"):
        cams_2 = read_mvsnet_cams(ARGS.data_path_2)
    elif(ARGS.format_2 == "colmap"):
        cams_2 = read_colmap_cams(ARGS.data_path_2)
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
