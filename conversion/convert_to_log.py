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

parse.add_argument("-d", "--data_path", default="/data/dtu/cams", type=str, help="Path to the camera datafor the given format.")
parse.add_argument("-f", "--format", default="mvsnet", type=str, help="The format type for the first stored camera data (e.g. mvsnet, colmap, ...).")
parse.add_argument("-a", "--alignment", default=None, type=str, help="The path to an alignment transformation.")
parse.add_argument("-o", "--output_file", default="./data/default_transform.txt", type=str, help="The path to the output file to store the transformation.")

ARGS = parse.parse_args()

def convert_to_log(cams, output_file, alignment):
    num_cams = len(cams)

    # write cameras into .log file
    with open(output_file, 'w') as f:
        for i,cam in enumerate(cams):
            # apply alignment transformation
            cam = alignment @ cam

            # write camera to output_file
            f.write("{} {} 0\n".format(str(i),str(i)))
            for row in cam:
                for c in row:
                    f.write("{} ".format(str(c)))
                f.write("\n")
        
    return

def main():
    # get the cameras for the given format
    if (ARGS.format == "mvsnet"):
        cams = load_mvsnet_cams(ARGS.data_path)
    else:
        print("ERROR: unknown format type '{}'".format(form))
        sys.exit()

    # read alignment file
    if(ARGS.alignment == None):
        A = np.eye(4)
    else:
        A = read_matrix(ARGS.alignment)

    # convert cameras into .log file format
    convert_to_log(cams, ARGS.output_file, A)

if __name__=="__main__":
    main()
