import os
import sys
import numpy as np
import cv2
from scipy.linalg import null_space

def load_mvsnet_cams(data_path):
    cam_files = os.listdir(data_path)
    cam_files.sort()

    cams = []
    
    for cf in cam_files:
        if (cf[-7:] != "cam.txt"):
            continue

        cam_path = os.path.join(data_path,cf)
        with open(cam_path,'r') as f:
            cam = np.zeros((4, 4))
            words = f.read().split()

            # read extrinsic
            for i in range(0, 4):
                for j in range(0, 4):
                    extrinsic_index = 4 * i + j + 1
                    cam[i][j] = float(words[extrinsic_index])
                    
            cams.append(cam)

    return cams

def load_colmap_cams(data_path):
    cam_file = os.path.join(data_path,"camera_poses.log")

    cams = []
    
    with open(cam_file,'r') as f:
        lines = f.readlines()

        for i in range(0,len(lines),5):
            cam = np.zeros((4, 4))
            # read extrinsic
            for j in range(1, 5):
                cam[j-1,:] = np.array([float(l.strip()) for l in lines[i+j].split()])
            cam = np.linalg.inv(cam)
                
            cams.append(cam)

    return cams

def read_matrix(trans_file):
    with open(trans_file, 'r') as f:
        lines = f.readlines()
        M = []

        for l in lines:
            row = l.split()
            row = [float(s) for s in row]
            M.append(row)
        M = np.array(M)

    return M

def write_matrix(M, filename):
    with open(filename, "w") as f:
        for row in M:
            for e in row:
                f.write("{} ".format(e))
            f.write("\n")

def camera_center(cam):
    C = null_space(cam[:3,:4])
    C /= C[3,:]

    return C

def build_pyr_point_cloud(pyr_pts, filename):
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
