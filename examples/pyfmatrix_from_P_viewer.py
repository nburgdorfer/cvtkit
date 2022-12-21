import numpy as np
import cv2
import skimage.transform as transform
import sys
import skimage

from cvt.visualization.util import *


def main():
    if len(sys.argv) < 6:
        print("Usage: pyfmatrix_from_P_viewer.py img1_path img2_path K_path P1_path P2_path [scale]")
        print("Examples:")
        print("   pyfmatrix_from_P_viewer.py data/264.bmp data/435.bmp data/K_tnt.txt data/P-264.txt data/P-435.txt")
        print("   pyfmatrix_from_P_viewer.py data/264.bmp data/435.bmp data/K_tnt.txt data/P-264.txt data/P-435.txt 0.25")
        quit(-1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    K_path = sys.argv[3]
    P1_path = sys.argv[4]
    P2_path = sys.argv[5]
    scale = 0.5
    if len(sys.argv) > 6:
        scale = float(sys.argv[6])

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    K =  np.loadtxt(K_path)
    K = np.reshape(K,(3,3))
    P1 = np.loadtxt(P1_path)
    P1 = np.reshape(P1,(3,4)) # 4x4 can be ignored later
    P2 = np.loadtxt(P2_path)
    P2 = np.reshape(P2,(3,4))

    f = fundamentalFromKP(K,P1,P2)

    fmat_demo(img1, img2, f, scale)

if __name__ == "__main__":
    main()
