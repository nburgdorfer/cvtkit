import sys, os
import numpy
from scipy.spatial.transform import Rotation as R
from cvt.io import read_cams_sfm

def gen_cameras_file(cams, output_path, cam_file="cameras.txt"):
    cam_str = "# Camera list with one line of data per camera:\n"
    cam_str += "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
    cam_str += "# Number of cameras: 1\n"
    cam_str += f"1 PINHOLE 1920 1080 {cams[0,1,0,0]} {cams[0,1,1,1]} {cams[0,1,0,2]} {cams[0,1,1,2]}\n"

    with open(os.path.join(output_path,cam_file),'w') as of:
        of.write(cam_str)

def gen_images_file(cams, image_files, output_path, image_file="images.txt"):
    img_str = "# Image list with two lines of data per image:\n"
    img_str += "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    img_str += "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
    img_str += "# Number of images: 251, mean observations per image: 4235.1593625498008\n"
    for i, img in enumerate(image_files):
        qx,qy,qz,qw = (R.from_matrix(cams[i,0,:3,:3])).as_quat()
        tx,ty,tz = cams[i,0,:3,3]
        img_str += f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img}\n\n"

    with open(os.path.join(output_path,image_file),'w') as of:
        of.write(img_str)

def main():
    cam_path = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[3]

    # read in cameras
    cams = read_cams_sfm(cam_path)

    # create cameras file
    gen_cameras_file(cams, output_path)

    # create images file
    image_files = os.listdir(image_path)
    image_files.sort()
    gen_images_file(cams, image_files, output_path)

    # create points3D file
    fp = open(os.path.join(output_path,"points3D.txt"),'w')
    fp.close()


if __name__=="__main__":
    main()
