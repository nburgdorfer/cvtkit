import cv2
import os
import sys

image_folder = sys.argv[1]
video_name = sys.argv[2]

image_files = os.listdir(image_folder)
image_files.sort()

images = [img for img in image_files if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, fps=8, frameSize=(width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
