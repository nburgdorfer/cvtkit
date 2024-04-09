import sys, os
import numpy as np
import cv2


video_file = sys.argv[1]
images_path = os.path.join(sys.argv[2], "images")
desired_frame_rate = 1


cam = cv2.VideoCapture(video_file)
fps = cam.get(cv2.CAP_PROP_FPS)

if not os.path.exists(images_path):
    os.makedirs(images_path)

sample_freq = int(fps//desired_frame_rate)
output_frame_ind = 0
input_frame_ind = 0
while(True): 
    # reading from frame 
    ret,frame = cam.read() 
    if ret: 
        if (input_frame_ind % sample_freq == 0):
            filename = os.path.join(images_path, f"{output_frame_ind:08d}.png")
            print(f"Creating image {filename}")
            cv2.imwrite(filename, frame)
            output_frame_ind += 1
        input_frame_ind += 1
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 
