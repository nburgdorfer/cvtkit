import os
import sys
import numpy as np
import cv2
import re
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation

def camera_center(cam):
    C = null_space(cam[:3,:4])
    C /= C[3,:]

    return C

def compute_relative_transform(cams_1, cams_2, A):
    centers_1 = np.squeeze(np.array([ camera_center(c) for c in cams_1 ]), axis=2)
    centers_2 = np.squeeze(np.array([ camera_center(c) for c in cams_2 ]), axis=2)

    ### determine scale
    # grab first camera pair
    c1_0 = centers_1[0][:3]
    c2_0 = centers_2[0][:3]

    # grab one-hundreth camera pair
    c1_1 = centers_1[99][:3]
    c2_1 = centers_2[99][:3]

    # calculate the baseline between both sets of cameras
    baseline_1 = np.linalg.norm(c1_0 - c1_1)
    baseline_2 = np.linalg.norm(c2_0 - c2_1)

    # compute the scale based on the baseline ratio
    scale = baseline_2/baseline_1

    ### determine 1->2 Rotation 
    b1 = np.array([c[:3] for c in centers_1])
    b2 = np.array([c[:3] for c in centers_2])
    R = Rotation.align_vectors(b2,b1)[0].as_matrix()
    R = scale * R

    ### create transformation matrix
    M = np.eye(4)
    M[:3,:3] = R

    ### determine 1->2 Translation
    num_cams = len(cams_1)
    t = np.array([ c2-(M@c1) for c1,c2 in zip(centers_1,centers_2) ])
    t = np.mean(t, axis=0)

    ### add translation
    M[:3,3] = t[:3]

    ### apply additional alignment
    M = A@M

    return M


def convert_to_log(cams, output_file, alignment):
    num_cams = len(cams)

    # write cameras into .log file
    with open(output_file, 'w') as f:
        for i,cam in enumerate(cams):
            # apply alignment transformation
            cam = alignment @ cam

            # write camera to output_file
            f.write("{} {} 0\n".format(str(i),str(i)))
            for row in cam:
                for c in row:
                    f.write("{} ".format(str(c)))
                f.write("\n")
        
    return

def convert_from_log(log_file, output_path, old_cam_path):
    # write cameras into .log file
    with open(log_file, 'r') as f:
        lines = f.readlines()
        num_lines = len(lines)
        i = 0

        while(i < num_lines-5):
            view_num = int(lines[i].strip().split(" ")[0])
            
            cam = np.zeros((2,4,4))
            cam[0,0,:] = np.asarray(lines[i+1].strip().split(" "), dtype=float)
            cam[0,1,:] = np.asarray(lines[i+2].strip().split(" "), dtype=float)
            cam[0,2,:] = np.asarray(lines[i+3].strip().split(" "), dtype=float)
            cam[0,3,:] = np.asarray(lines[i+4].strip().split(" "), dtype=float)
            cam[0,:,:] = np.linalg.inv(cam[0,:,:])

            cam_file = "{:08d}_cam.txt".format(view_num)
            old_cam  = load_cam(os.path.join(old_cam_path,cam_file), 256)
            new_cam_path = os.path.join(output_path, cam_file)

            cam[1,:,:] = old_cam[1,:,:]

            write_cam(new_cam_path, cam)
            i = i+5
    return



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
