import cv2
import os
import sys
import numpy as np

image_folder = sys.argv[1]
depth_folder = sys.argv[2]
video_name = sys.argv[3]
height = 600
width = 1600

image_files = os.listdir(image_folder)
image_files.sort()

depth_files = os.listdir(depth_folder)
depth_files.sort()

#images = [img for img in image_files if img.endswith(".png") and (img in depth_files)]
#depths = [depth for depth in depth_files if depth.endswith(".png")]
images = [img for img in image_files if img.endswith(".png") ]
depths = [depth for depth in depth_files if depth.endswith(".png")]

assert(len(images) == len(depths))

video = cv2.VideoWriter(video_name, 0, fps=8, frameSize=((width,height)))

for (image,depth) in zip(images, depths):
    img = cv2.imread(os.path.join(image_folder, image))

    image_name = image[:-4]
    dp = cv2.imread(os.path.join(depth_folder, depth))
    frame = np.concatenate((img,dp), axis=1)
    frame = cv2.putText(frame, image_name, (50,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (49, 73, 250), 1)

    video.write(frame)

cv2.destroyAllWindows()
video.release()
