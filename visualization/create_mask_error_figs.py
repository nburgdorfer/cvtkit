import sys
import os
import numpy as np
import argparse
import cv2

from typing import List

from create_latex_figures import create_subfigures

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
        if (f[:12] == "depth_visual"):
            gt_files.append(os.path.join(gt_path,f))

    # grab image file names
    files = os.listdir(img_path)
    files.sort()

    for f in files:
        img_files.append(os.path.join(img_path,f))

    return np.array(bmask_files), np.array(gt_files), np.array(img_files)

def compute_error_masks(bmask_files: List[str], gt_depth_files: List[str], output_path: str = "./output/"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_files = []

    for i, (bmask,gt_depth) in enumerate(zip(bmask_files, gt_depth_files)):
        gt_img = cv2.imread(gt_depth)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        rows,cols = gt_img.shape

        bm_img = cv2.imread(bmask)
        bm_img = cv2.cvtColor(bm_img, cv2.COLOR_BGR2GRAY)
        bm_img = cv2.resize(bm_img, dsize=(cols,rows))

        # compute accuracy error
        bm = np.greater(bm_img,0.0).reshape(rows,cols,1)
        gt = np.less_equal(gt_img,0.0).reshape(rows,cols,1)

        ones = np.ones((rows,cols,3))
        red = np.tile(np.array([0,0,255]).reshape(1,1,3), (rows,cols,1))
        acc_error = red * ((bm*gt) * ones)

        #cv2.imwrite("acc_error.png", acc_error)

        # compute completeness error
        bm = np.less_equal(bm_img,0.0).reshape(rows,cols,1)
        gt = np.greater(gt_img,0.0).reshape(rows,cols,1)

        ones = np.ones((rows,cols,3))
        yellow = np.tile(np.array([0,255,255]).reshape(1,1,3), (rows,cols,1))
        comp_error = yellow * ((bm*gt) * ones)

        #cv2.imwrite("comp_error.png", comp_error)

        # compile both into one figure
        error = acc_error + comp_error

        output_file = os.path.join(output_path, "{:03d}_error.png".format(i))
        cv2.imwrite(output_file, error)

        output_files.append(output_file)

    return np.array(output_files)


def build_latex_doc(data):
    subfigs, views = data.shape

    latex_doc_str = "\\documentclass{article}\n" + \
                    "\\usepackage[utf8]{inputenc}\n" + \
                    "\\usepackage{graphicx}\n" + \
                    "\\usepackage{subcaption}\n" + \
                    "\\graphicspath{ {../} }\n" + \
                    "\\begin{document}\n"

    for v in range(views):
        figure_data = []

        for s in range(subfigs):
            figure_data.append(data[s,v])

        captions = ["error","ground truth","image","Analysis of Mask Error for View {}".format(v)]
        labels = ["error{}".format(v),"gt{}".format(v),"img{}".format(v),"error_analys{}".format(v)]
        latex_fig = create_subfigures(figure_data, captions, labels)

        #print(latex_fig)
        latex_doc_str += latex_fig

    latex_doc_str += "\\end{document}\n"

    return latex_doc_str

def main():
    bmask_files, gt_depth_files, img_files = load_files(ARGS.bmask_data_path, ARGS.gt_data_path, ARGS.img_data_path)

    error_mask_files = compute_error_masks(bmask_files, gt_depth_files)

    data = np.vstack((error_mask_files, gt_depth_files, img_files))

    document = build_latex_doc(data)

    with open(ARGS.output_file, 'w') as f:
        f.write(document)

if __name__=="__main__":
    main()
