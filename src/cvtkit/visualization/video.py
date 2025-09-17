# cvt/visualization/video.py
"""Module including routines for creating videos.

This module includes the following functions:

- `video_from_images(image_files, video_file, frame_rate)` - Creates a video from a set of images.
- `video_from_images2(image1_files, image2_files, video_file, frame_rate, orientation)` - Creates a video from two sets of images, stitching the images together each frame.
"""
import cv2
import os
import sys
from typing import List


def video_from_images(
    image_files: List[str], video_file: str, frame_rate: int = 15
) -> None:
    """Creates a video from a set of images.

    Parameters:
        image_files: List of image files to be stitched into a video.
        video_file: Output video file to be created.
        frame_rate: Desired frame rate of the video.
    """
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_file, 0, fps=frame_rate, frameSize=(width, height))

    for image_file in image_files:
        video.write(cv2.imread(image_file))

    cv2.destroyAllWindows()
    video.release()

    return


def video_from_images2(
    image1_files: List[str],
    image2_files: List[str],
    video_file: str,
    frame_rate: int = 15,
    orientation: str = "horizontal",
) -> None:
    """Creates a video from two sets of images, stitching the images together each frame.

    Parameters:
        image1_files: List of first image files to be stitched.
        image2_files: List of second image files to be stitched.
        video_file: Output video file to be created.
        frame_rate: Desired frame rate of the video.
        orientation: Desired stitch orientation (horizontal or verticle).
    """
    frame1 = cv2.imread(image1_files[0])
    height1, width1, layers1 = frame1.shape

    frame2 = cv2.imread(image2_files[0])
    height2, width2, layers2 = frame2.shape

    assert len(image1_files) == len(image2_files)
    assert height1 == height2
    assert width1 == width2
    assert layers1 == layers2

    if orientation == "horizontal":
        video = cv2.VideoWriter(
            video_file, 0, fps=frame_rate, frameSize=((width * 2, height))
        )
        for img1_file, img2_file in zip(image1_files, image2_files):
            img1 = cv2.imread(img1_file)
            img2 = cv2.imread(img2_file)

            frame = np.concatenate((img1, img2), axis=1)
            video.write(frame)
        cv2.destroyAllWindows()
        video.release()

    else:
        print(f"Orientation '{orientation}' is not yet supported.")

    return
