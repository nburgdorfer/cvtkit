import sys
import os
from tqdm import tqdm
import argparse

import numpy as np
import cv2

from cvt.io import read_mesh, read_cams_sfm
from cvt.geometry import project_renderer

# argument parsing
parse = argparse.ArgumentParser(description="Video->Frames generator.")
parse.add_argument("-o", "--output_path", default="images/", type=str,
        help="Output directory where all frames of the video will be stored.")
parse.add_argument("-v", "--video_file", default="", type=str,
        help="The name of the output video file (with extension). This will be stored in the output path (ex: mesh_vid.avi)")
ARGS = parse.parse_args()

def main():
    if not os.path.exists(ARGS.output_path):
        os.mkdir(ARGS.output_path)

    cap = cv2.VideoCapture(ARGS.video_file)
    image_ind = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                print(f"Writing image for frame {image_ind:08d}")
                img_file = os.path.join(ARGS.output_path, f"{image_ind:08d}.png")
                cv2.imwrite(img_file, frame)
                image_ind += 1
        if cv2.waitKey(10) == 27:
            break

if __name__=="__main__":
    main()
