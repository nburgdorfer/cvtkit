# cvt/visualization/util.py
"""Module including general utilities for visualization.

This module includes the following functions:


"""
import open3d as o3d
import numpy as np
import cv2
import scipy.ndimage as ndimage
import skimage.transform as transform

from io import *

def display_inlier_outlier(cloud: o3d.geometry.PointCloud, indices: np.ndarray) -> None:
    """Displays a point cloud with outlier points colored red.

    Parameters:
        cloud: Point cloud to be displayed.
        indices: Indices indicating the inlier points.
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

    return

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
    img_scaled = transform.resize(img, [img.shape[0]*scale, img.shape[1]*scale], mode="constant")
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

    cv2.namedWindow("img1")
    cv2.namedWindow("img2")
    cv2.setMouseCallback("img1", mouse1_callback)

    while True:
        show2 = draw_line(img2, img2_line)
        show1 = img1
        cv2.imshow("img1", cv2.cvtColor(show1, cv2.COLOR_BGR2RGB))
        cv2.imshow("img2", cv2.cvtColor(show2, cv2.COLOR_BGR2RGB))
        cv2.waitKey(50)

def build_pyr_point_cloud(pyr_pts, filename):
    """Builds a point cloud for a camera frustum visual.

    !!! bug "needs work..."
    """
    num_pts = len(pyr_pts)
    element_vertex = 6*num_pts
    element_edge = 10*num_pts

    with open(filename, 'w') as fh:
        # write header meta-data
        fh.write('ply\n')
        fh.write('format ascii 1.0\n')
        fh.write('comment Right-Handed System\n')
        fh.write('element vertex {}\n'.format(element_vertex))
        fh.write('property float x\n')
        fh.write('property float y\n')
        fh.write('property float z\n')
        fh.write('property uchar red\n')
        fh.write('property uchar green\n')
        fh.write('property uchar blue\n')
        fh.write('element edge {}\n'.format(element_edge))
        fh.write('property int vertex1\n')
        fh.write('property int vertex2\n')
        fh.write('property uchar red\n')
        fh.write('property uchar green\n')
        fh.write('property uchar blue\n')
        fh.write('end_header\n')

        # write vertex data to file
        for pt in pyr_pts:
            fh.write('{:.10f}'.format( pt[0][0][0] ) + ' ' + '{:.10f}'.format( pt[0][1][0] ) + ' ' + '{:.10f}'.format( pt[0][2][0] ) + ' 255 128 0\n')
            fh.write('{:.10f}'.format( pt[1][0][0] ) + ' ' + '{:.10f}'.format( pt[1][1][0] ) + ' ' + '{:.10f}'.format( pt[1][2][0] ) + ' 255 128 0\n')
            fh.write('{:.10f}'.format( pt[2][0][0] ) + ' ' + '{:.10f}'.format( pt[2][1][0] ) + ' ' + '{:.10f}'.format( pt[2][2][0] ) + ' 255 128 0\n')
            fh.write('{:.10f}'.format( pt[3][0][0] ) + ' ' + '{:.10f}'.format( pt[3][1][0] ) + ' ' + '{:.10f}'.format( pt[3][2][0] ) + ' 255 128 0\n')
            fh.write('{:.10f}'.format( pt[4][0][0] ) + ' ' + '{:.10f}'.format( pt[4][1][0] ) + ' ' + '{:.10f}'.format( pt[4][2][0] ) + ' 255 128 0\n')
            fh.write('{:.10f}'.format( pt[5][0][0] ) + ' ' + '{:.10f}'.format( pt[5][1][0] ) + ' ' + '{:.10f}'.format( pt[5][2][0] ) + ' 255 128 0\n')

        # write edge data to file
        for i in range(num_pts):
            edge_ind = i*6
            fh.write('{} {} 255 0 0\n'.format(edge_ind, edge_ind+1))
            fh.write('{} {} 255 0 0\n'.format(edge_ind, edge_ind+2))
            fh.write('{} {} 255 0 0\n'.format(edge_ind, edge_ind+3))
            fh.write('{} {} 255 0 0\n'.format(edge_ind, edge_ind+4))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+1, edge_ind+2))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+2, edge_ind+3))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+3, edge_ind+4))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+4, edge_ind+1))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+1, edge_ind+5))
            fh.write('{} {} 255 0 0\n'.format(edge_ind+5, edge_ind+2))
    return

def build_cam_pyr(cam_scale, K):
    """Constructs a camera frustum for visualization.

    !!! bug "needs work..."
    """
    focallen   = K[0][0]
    cam_w      = 2 * K[0][2]
    cam_h      = 2 * K[1][2]
    cam_center = np.array([0.0,          0.0,          0.0,      1.0])
    cam_ul     = np.array([cam_w * -0.5, cam_h * -0.5, focallen, 1.0])
    cam_ur     = np.array([cam_w *  0.5, cam_h * -0.5, focallen, 1.0])
    cam_dr     = np.array([cam_w *  0.5, cam_h *  0.5, focallen, 1.0])
    cam_dl     = np.array([cam_w * -0.5, cam_h *  0.5, focallen, 1.0])
    cam_top    = np.array([0.0,          cam_h * -0.7, focallen, 1.0])
    cam_center *= cam_scale
    cam_ul     *= cam_scale
    cam_ur     *= cam_scale
    cam_dr     *= cam_scale
    cam_dl     *= cam_scale
    cam_top    *= cam_scale
    cam_center[3] = 1.0
    cam_ul[3]     = 1.0
    cam_ur[3]     = 1.0
    cam_dr[3]     = 1.0
    cam_dl[3]     = 1.0
    cam_top[3]    = 1.0
    cam_center = cam_center.reshape((4, 1))
    cam_ul     = cam_ul.reshape((4, 1))
    cam_ur     = cam_ur.reshape((4, 1))
    cam_dr     = cam_dr.reshape((4, 1))
    cam_dl     = cam_dl.reshape((4, 1))
    cam_top    = cam_top.reshape((4, 1))
    return [cam_center, cam_ul, cam_ur, cam_dr, cam_dl, cam_top]


def display_map(filename: str, disp_map: np.ndarray, mx: float, mn: float) -> None:
    """Writes an input map to a normalized image.

    Parameters:
        filename: Name of the file to store the input map.
        disp_map: Map to be stored as an image file.
        mx: maximum value used for pixel intensity normalization.
        mn: minimum value used for pixel intensity normalization.
    """
    disp_map = ((disp_map-mn)/(mx-mn+1e-8))*255
    cv2.imwrite(filename, disp_map)



