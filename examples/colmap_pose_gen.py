import sys, os
import numpy
from scipy.spatial.transform import Rotation as R
from cvt.io import read_cams_sfm
import cv2

def gen_cameras_file(cams, h, w, output_path, cam_file="cameras.txt"):
    cam_str = "# Camera list with one line of data per camera:\n"
    cam_str += "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
    cam_str += "# Number of cameras: 1\n"
    cam_str += f"1 PINHOLE {w} {h} {cams[0,1,0,0]} {cams[0,1,1,1]} {cams[0,1,0,2]} {cams[0,1,1,2]}\n"

    with open(os.path.join(output_path,cam_file),'w') as of:
        of.write(cam_str)

def gen_images_file(cams, image_files, init_colmap_path, output_path, image_file="images.txt"):
    with open(os.path.join(init_colmap_path, "images.txt"), 'r') as icf:
        lines = icf.readlines()[4::2]
        database_ids = {}
        for line in lines:
            line = line.strip().split()
            database_ids[f"{line[0]}"] = line[-1]

        img_str = "# Image list with two lines of data per image:\n"
        img_str += "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        img_str += "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        img_str += "# Number of images: , mean observations per image: \n"
        for ind in range(1, len(image_files)+1):
            img = database_ids[f"{ind}"]
            i = int(img[:-4])
            qx,qy,qz,qw = (R.from_matrix(cams[i,0,:3,:3])).as_quat()
            tx,ty,tz = cams[i,0,:3,3]
            img_str += f"{ind} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img}\n\n"
    with open(os.path.join(output_path,image_file),'w') as of:
        of.write(img_str)

def main():
    cam_path = sys.argv[1]
    init_colmap_path = sys.argv[2]
    image_path = sys.argv[3]
    output_path = sys.argv[4]

    # read in data
    cams = read_cams_sfm(cam_path)
    image_files = os.listdir(image_path)
    image_files.sort()

    # create cameras file
    img = cv2.imread(os.path.join(image_path,image_files[0]))
    h,w,_ = img.shape
    gen_cameras_file(cams, h, w, output_path)

    # create images file
    gen_images_file(cams, image_files, init_colmap_path, output_path)

    # create points3D file
    fp = open(os.path.join(output_path,"points3D.txt"),'w')
    fp.close()


if __name__=="__main__":
    main()
