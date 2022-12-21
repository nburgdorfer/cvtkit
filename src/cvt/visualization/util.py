import open3d as o3d
import numpy as np
from common.io import *
import cv2
import scipy.ndimage as ndimage
import skimage.transform as transform

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def plot_cameras(cams, num_cams, scale, A, output_file):
    # grab the requested number of cameras and apply the alignment
    P = cams[:num_cams]
    P = np.array([ A @ np.linalg.inv(p) for p in P ])

    # create 'num_cams' intrinsic matrices (just used for point-cloud camera pyramid geometry)
    k = np.array([[233.202, 0.0, 144.753],[0.0, 233.202, 108.323],[0.0, 0.0, 1.0]])
    K = np.array([ k for p in P])

    # build list of camera pyramid points
    pyr_pts = []
    for k,p in zip(K,P):
        pyr_pt = build_cam_pyr(scale, k)
        pyr_pt = p @ pyr_pt
        pyr_pts.append(pyr_pt)

    # build point cloud using camera centers
    build_pyr_point_cloud(pyr_pts, output_file)

    return


f_matrix = None
img2_line = None
img1 = None
img2 = None

CENTERED = False


def mouse1_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        #print("mouse1 X: " + str(x) + " Y: " + str(y))
#        y = img1.shape[0] - y
        if CENTERED:
            x -= img1.shape[1] / 2
            y -= img1.shape[0] / 2
        mouse1_pt = np.asarray([x, y, 1.0])
        i2ray = np.dot(f_matrix, mouse1_pt)
        if CENTERED:
            i2ray[2] = i2ray[2] - i2ray[0]*img2.shape[1]/2 - i2ray[1]*img2.shape[0]/2 
        #print(i2ray)
        i2pt1_x = 0
        i2pt2_x = img2.shape[1]
        i2pt1_y = int(-(i2ray[2] + i2ray[0] * i2pt1_x) / i2ray[1])
        i2pt2_y = int(-(i2ray[2] + i2ray[0] * i2pt2_x) / i2ray[1])

        global img2_line
#        img2_line = ((i2pt1_x, img1.shape[0] - i2pt1_y), (i2pt2_x, img1.shape[0] - i2pt2_y))
        img2_line = ((i2pt1_x, i2pt1_y), (i2pt2_x, i2pt2_y))

def draw_line(img, line):
    if line is None:
        return img
    # Clip the line to image bounds
    ret, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), line[0], line[1])
    if ret:
        img = img.copy()
        cv2.line(img, p1, p2, (255, 255, 0), 1)
    return img


def scale_img(img, scale):
    # This function takes an integer image in range of 0-255
    # and returns a float64 image in range of 0-1
    img_scaled = transform.resize(img,
                    [img.shape[0] * scale, img.shape[1] * scale], 
                    mode="constant")
    # Cast it to float32 to appease OpenCV later on
    return img_scaled.astype(np.float32)


def scale_f_mat(mat, scale):
    mat[:, 2] *= scale
    mat[2, :] *= scale
    return mat


def fmat_demo(img1l, img2l, fl, scale=1.0):
    global img1, img2, f_matrix

    img1 = scale_img(img1l, scale)
    img2 = scale_img(img2l, scale)
    f_matrix = scale_f_mat(fl, scale)

    f_matrix = f_matrix/np.max(f_matrix)

    print("Img1: " + str(img1.shape))
    print("Img2: " + str(img2.shape))
    #print(f)

    cv2.namedWindow("img1")
    cv2.namedWindow("img2")
    cv2.setMouseCallback("img1", mouse1_callback)

    while True:
        show2 = draw_line(img2, img2_line)
        show1 = img1
        cv2.imshow("img1", cv2.cvtColor(show1, cv2.COLOR_BGR2RGB))
        cv2.imshow("img2", cv2.cvtColor(show2, cv2.COLOR_BGR2RGB))
        cv2.waitKey(50)
