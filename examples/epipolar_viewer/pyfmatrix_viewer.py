import numpy as np
import cv2
import scipy.ndimage as ndimage
import skimage.transform as transform
import sys

from cvtkit.visualization.util import *

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: pyfmatrix_viewer.py img1_path img2_path F_path [scale]")
        print("Examples:")
        print("   pyfmatrix_viewer.py data/264.bmp data/435.bmp data/f-264-435.txt")
        print("   pyfmatrix_viewer.py data/264.bmp data/435.bmp data/f-264-435.txt 0.25")
        quit(-1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    f_path = sys.argv[3]
    scale = 1.0
    if len(sys.argv) > 4:
        scale = float(sys.argv[4])

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    f = np.loadtxt(f_path)

    fmat_demo(img1, img2, f, scale)
