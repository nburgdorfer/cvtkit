import os
import argparse
from tqdm import tqdm
import sys
import cv2
import numpy as np
import imageio.v2 as iio

def generate_gif_1(vid_file, image_path, fps):
    image_files = os.listdir(image_paths)
    image_files.sort()
    with iio.get_writer(vid_file, format="GIF-PIL", mode="I", fps=fps, loop=0) as writer:
        for imf in tqdm(image_files, desc="Generating GIF", unit="frames"):
            image = cv2.imread(os.path.join(image_path, imf))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            writer.append_data(image)


def generate_video_1(vid_file, image_path, fps):
    image_files = os.listdir(image_path)
    image_files.sort()
    image = cv2.imread(os.path.join(image_path, image_files[0]))
    h,w,_ = image.shape

    with iio.get_writer(vid_file, format="FFMPEG", mode="I", fps=fps) as writer:
        for imf in tqdm(image_files, desc="Generating Video", unit="frames"):
            image = cv2.imread(os.path.join(image_path, imf))
            image = cv2.resize(image, (w,h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            writer.append_data(image)

def main():
    if (len(sys.argv) != 4):
        print(f"Error: usage python {sys.argv[0]} <image-path> <video-file-name> <fps>")
        sys.exit()


    image_path = sys.argv[1]
    vid_file = sys.argv[2]
    fps = min(int(sys.argv[3]), 144)

    if (vid_file[-3:]==".gif"):
        generate_gif_1(vid_file, image_path, fps)
    else:
        generate_video_1(vid_file, image_path, fps)


if __name__=="__main__":
    main()
