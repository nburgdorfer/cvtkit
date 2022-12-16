import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys
import os
import argparse
import scipy.io as sio


# argument parsing
parse = argparse.ArgumentParser(description="Point Cloud Comparison Tool.")

parse.add_argument("-r", "--src_ply", default="../src.ply", type=str, help="Path to source point cloud file.")
parse.add_argument("-t", "--tgt_ply", default="../tgt.ply", type=str, help="Path to target point cloud file.")
parse.add_argument("-o", "--output_path", default="./evaluation", type=str, help="Output path where all metrics and results will be stored.")
parse.add_argument("-v", "--voxel_size", default=0.2, type=float, help="Voxel size used for consistent downsampling.")
parse.add_argument("-x", "--max_dist", default=0.4, type=float, help="Max distance threshold for point matching.")
parse.add_argument("-k", "--mask_th", default=20, type=float, help="Masking threshold to remove outliers from comparison.")

ARGS = parse.parse_args()


def correct_round(n):
    return np.round(n+0.5)

def read_point_cloud(ply_path, size=0.1):
    if(ply_path[-3:] != "ply"):
        print("{} is not a '.ply' file.".format(ply_path))
        sys.exit()

    ply = o3d.io.read_point_cloud(ply_path)
    ply = ply.voxel_down_sample(voxel_size=size)

    return ply

def build_src_points_filter(ply, min_bound, res, mask):
    points = np.asarray(ply.points).transpose()
    shape = points.shape
    mask_shape = mask.shape
    filt = np.zeros(shape[1])

    min_bound = min_bound.reshape(3,1)
    min_bound = np.tile(min_bound, (1,shape[1]))

    qv = points
    qv = (points - min_bound) / res
    qv = correct_round(qv).astype(int)

    # get all valid points
    in_bounds = np.asarray(np.where( ((qv[0,:]>=0) & (qv[0,:] < mask_shape[0]) & (qv[1,:]>=0) & (qv[1,:] < mask_shape[1]) & (qv[2,:]>=0) & (qv[2,:] < mask_shape[2])))).squeeze(0)
    valid_points = qv[:,in_bounds]

    # convert 3D coords ([x,y,z]) to appropriate flattened coordinate ((x*mask_shape[1]*mask_shape[2]) + (y*mask_shape[2]) + z )
    mask_inds = np.ravel_multi_index(valid_points, dims=mask.shape, order='C')

    # further trim down valid points by mask value (keep point if mask is True)
    mask = mask.flatten()
    valid_mask_points = np.asarray(np.where(mask[mask_inds] == True)).squeeze(0)

    # add 1 to indices where we want to keep points
    filt[in_bounds[valid_mask_points]] = 1

    return filt

def build_tgt_points_filter(ply, P):
    points = np.asarray(ply.points).transpose()
    shape = points.shape

    # compute iner-product between points and the defined plane
    Pt = P.transpose()

    points = np.concatenate((points, np.ones((1,shape[1]))), axis=0)
    plane_prod = (Pt @ points).squeeze(0)

    # get all valid points
    filt = np.asarray(np.where((plane_prod > 0), 1, 0))

    return filt

def compare_point_clouds(src_ply, tgt_ply, mask_th, max_dist):
    # compute bi-directional distance between point clouds
    dists_src = np.asarray(src_ply.compute_point_cloud_distance(tgt_ply))
    valid_dists = np.where(dists_src <= mask_th)[0]
    dists_src = dists_src[valid_dists]

    dists_tgt = np.asarray(tgt_ply.compute_point_cloud_distance(src_ply))
    valid_dists = np.where(dists_tgt <= mask_th)[0]
    dists_tgt = dists_tgt[valid_dists]

    # compute accuracy and competeness
    acc = np.mean(dists_src)
    comp = np.mean(dists_tgt)

    # measure incremental precision and recall values with thesholds from (0, 10*max_dist)
    th_vals = np.linspace(0, 3*max_dist, num=50)
    prec_vals = [ (len(np.where(dists_src <= th)[0]) / (len(dists_src)+1e-5)) for th in th_vals ]
    rec_vals = [ (len(np.where(dists_tgt <= th)[0]) / (len(dists_tgt)+1e-5)) for th in th_vals ]

    # compute precision and recall for given distance threshold
    prec = len(np.where(dists_src <= max_dist)[0]) / (len(dists_src)+1e-5)
    rec = len(np.where(dists_tgt <= max_dist)[0]) / (len(dists_tgt)+1e-5)

    # color point cloud for precision
    src_size = len(src_ply.points)
    cmap = plt.get_cmap("hot_r")
    colors = cmap(np.minimum(dists_src, max_dist) / max_dist)[:, :3]
    src_ply.colors = o3d.utility.Vector3dVector(colors)

    # color point cloud for recall
    tgt_size = len(tgt_ply.points)
    cmap = plt.get_cmap("hot_r")
    colors = cmap(np.minimum(dists_tgt, max_dist) / max_dist)[:, :3]
    tgt_ply.colors = o3d.utility.Vector3dVector(colors)

    return (src_ply, tgt_ply), (acc,comp), (prec, rec), (th_vals, prec_vals, rec_vals), (src_size, tgt_size)

def save_ply(file_path, ply):
    o3d.io.write_point_cloud(file_path, ply)

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

def main():
    ##### Initialization #####
    # set parameters
    src_path = ARGS.src_ply
    tgt_path = ARGS.tgt_ply
    output_path = ARGS.output_path
    voxel_size = ARGS.voxel_size
    max_dist = ARGS.max_dist
    mask_th = ARGS.mask_th

    # create output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    ##### Load in point clouds #####
    print("Loading point clouds...")
    src_ply = read_point_cloud(src_path, voxel_size)
    tgt_ply = read_point_cloud(tgt_path, voxel_size)


    ##### Compute metrics between point clouds #####
    print("Computing metrics between point clouds...")
    (precision_ply, recall_ply), (acc,comp), (prec, rec), (th_vals, prec_vals, rec_vals), (src_size, tgt_size) \
            = compare_point_clouds(src_ply, tgt_ply, mask_th, max_dist)


    ##### Save metrics #####
    print("Saving evaluation statistics...")
    # save precision point cloud
    precision_path = os.path.join(output_path, "precision.ply")
    save_ply(precision_path, precision_ply)

    # save recall point cloud
    recall_path = os.path.join(output_path, "recall.ply")
    save_ply(recall_path, recall_ply)

    # create plots for incremental threshold values
    plot_filename = os.path.join(output_path, "metrics.png")
    plt.plot(th_vals, prec_vals, th_vals, rec_vals)
    plt.title("Precision and Recall (t={}mm)".format(max_dist))
    plt.xlabel("threshold")
    plt.vlines(max_dist, 0, 1, linestyles='dashed', label='t')
    plt.legend(("precision", "recall"))
    plt.grid()
    plt.savefig(plot_filename)

    # write all metrics to the evaluation file
    stats_file = os.path.join(output_path, "evaluation_metrics.txt")
    with open(stats_file, 'w') as f:
        f.write("Voxel_size: {:0.3f}mm | Distance threshold: {:0.3f}mm\n".format(voxel_size, max_dist))
        f.write("Source point cloud size: {}\n".format(src_size))
        f.write("Target point cloud size: {}\n".format(tgt_size))
        f.write("Accuracy: {:0.3f}mm\n".format(acc))
        f.write("Completness: {:0.3f}mm\n".format(comp))
        f.write("Precision: {:0.3f}\n".format(prec))
        f.write("Recall: {:0.3f}\n".format(rec))


if __name__=="__main__":
    main()
