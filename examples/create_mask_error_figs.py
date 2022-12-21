import sys
import os
import numpy as np
import argparse
import cv2

from typing import List

from create_latex_figures import create_subfigures

# custom imports
from cvt.visualization.latex_util import *

# argument parsing
parse = argparse.ArgumentParser(description="Camera Plotting Tool.")

parse.add_argument("-b", "--bmask_data_path", default="/data/dtu/output", type=str, help="Path to the output binary mask data.")
parse.add_argument("-g", "--gt_data_path", default="/data/dtu/gt_depths", type=str, help="Path to the ground truth mask data.")
parse.add_argument("-i", "--img_data_path", default="/data/dtu/images", type=str, help="Path to the image data.")
parse.add_argument("-o", "--output_file", default="/data/figs.txt", type=str, help="The output latex file name/path.")

ARGS = parse.parse_args()

def load_files(bmask_path: str, gt_path: str, img_path: str):
    bmask_files = []
    gt_files = []
    img_files = []

    # grab output maks file names
    files = os.listdir(bmask_path)
    files.sort()

    for f in files:
        if (f[-14:] == "bmask_disp.png"):
            bmask_files.append(os.path.join(bmask_path,f))

    # grab gt depth file names
    files = os.listdir(gt_path)
    files.sort()

    for f in files:
        if (f[-9:] == "depth.pfm"):
            gt_files.append(os.path.join(gt_path,f))

    # grab image file names
    files = os.listdir(img_path)
    files.sort()

    for f in files:
        img_files.append(os.path.join(img_path,f))

    return np.array(bmask_files), np.array(gt_files), np.array(img_files)

def compute_error_masks(bmask_files: List[str], gt_depth_files: List[str], img_files: List[str], output_path: str = "./output/"):
    bmask_output_path = os.path.join(output_path, "bmasks")
    if not os.path.exists(bmask_output_path):
        os.makedirs(bmask_output_path)

    gt_output_path = os.path.join(output_path, "gt_depths")
    if not os.path.exists(gt_output_path):
        os.makedirs(gt_output_path)

    img_output_path = os.path.join(output_path, "images")
    if not os.path.exists(img_output_path):
        os.makedirs(img_output_path)

    error_output_path = os.path.join(output_path, "error")
    if not os.path.exists(error_output_path):
        os.makedirs(error_output_path)

    num_views = len(bmask_files)

    bmask_output_files = []
    gt_output_files = []
    img_output_files = []
    error_output_files = []
    offset = 9

    for v in range(num_views):
        bmask = bmask_files[v]
        gt_depth = gt_depth_files[v]
        img = img_files[v]

        # load gt depth maps
        with open(gt_depth, 'rb') as gtd:
            gt_img = load_pfm(gtd)
        #gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        rows,cols = gt_img.shape

        m = np.max(gt_img)
        g = (gt_img / m) * 255
        gt_output_file = os.path.join(gt_output_path, "{:03d}_gt.png".format(v))
        cv2.imwrite(gt_output_file, g)

        # load bmasks
        bm_img = cv2.imread(bmask)
        bm_img = cv2.cvtColor(bm_img, cv2.COLOR_BGR2GRAY)
        bm_img = cv2.resize(bm_img[9:-9,18:-18], dsize=(cols,rows))
        bmask_output_file = os.path.join(bmask_output_path, "{:03d}_bmask.png".format(v))
        cv2.imwrite(bmask_output_file, bm_img)

        # load images
        image = cv2.imread(img)
        image = cv2.resize(image[9:-9,18:-18], dsize=(cols,rows))
        img_output_file = os.path.join(img_output_path, "{:03d}_img.png".format(v))
        cv2.imwrite(img_output_file, image)

        # compute accuracy error
        bm = np.greater(bm_img,0.0).reshape(rows,cols,1)
        gt = np.less_equal(gt_img,0.0).reshape(rows,cols,1)

        ones = np.ones((rows,cols,3))
        red = np.tile(np.array([0,0,255]).reshape(1,1,3), (rows,cols,1))
        acc_error = red * ((bm*gt) * ones)

        # compute completeness error
        bm = np.less_equal(bm_img,0.0).reshape(rows,cols,1)
        gt = np.greater(gt_img,0.0).reshape(rows,cols,1)

        ones = np.ones((rows,cols,3))
        yellow = np.tile(np.array([0,255,255]).reshape(1,1,3), (rows,cols,1))
        comp_error = yellow * ((bm*gt) * ones)

        # compile both into one figure
        error = acc_error + comp_error

        error_output_file = os.path.join(error_output_path, "{:03d}_error.png".format(v))
        cv2.imwrite(error_output_file, error)

        gt_output_files.append(gt_output_file)
        bmask_output_files.append(bmask_output_file)
        img_output_files.append(img_output_file)
        error_output_files.append(error_output_file)

    return np.array(bmask_output_files), np.array(gt_output_files), np.array(img_output_files), np.array(error_output_files)

def main():
    bmask_files, gt_depth_files, img_files = load_files(ARGS.bmask_data_path, ARGS.gt_data_path, ARGS.img_data_path)

    bmask_output_files, gt_output_files, img_output_files, error_output_files = compute_error_masks(bmask_files, gt_depth_files, img_files)

    data = np.vstack((error_output_files, gt_output_files, img_output_files))

    document = build_latex_doc(data)

    with open(ARGS.output_file, 'w') as f:
        f.write(document)

if __name__=="__main__":
    main()
