import cv2
import os
import sys

def video_from_images(image_files, video_file, frame_rate=15):
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_file, 0, fps=frame_rate, frameSize=(width,height))

    for image_file in image_files:
        video.write(cv2.imread(image_file))

    cv2.destroyAllWindows()
    video.release()

    return

def video_from_images2(image1_files, image2_files, video_file, frame_rate=15, orientation="horizontal"):
    frame1 = cv2.imread(image1_files[0])
    height1, width1, layers1 = frame1.shape

    frame2 = cv2.imread(image2_files[0])
    height2, width2, layers2 = frame2.shape

    assert(len(image1_files) == len(image2_files))
    assert(height1==height2)
    assert(width1==width2)
    assert(layers1==layers2)

    if (orientation=="horizontal"):
        video = cv2.VideoWriter(video_file, 0, fps=frame_rate, frameSize=((width*2,height)))
        for (img1_file, img2_file) in zip(image1_files, image2_files):
            img1 = cv2.imread(img1_file)
            img2 = cv2.imread(img2_file)

            frame = np.concatenate((img1,img2), axis=1)
            video.write(frame)
        cv2.destroyAllWindows()
        video.release()

    else:
        print(f"Orientation '{orientation}' is not yet supported.")

    return

