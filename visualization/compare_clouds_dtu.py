import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys
import os
import argparse
import scipy.io as sio


# argument parsing
parse = argparse.ArgumentParser(description="Point Cloud Comparison Tool.")

parse.add_argument("-m", "--method", default="fusion", type=str, help="Method name (e.x. colmap).")
parse.add_argument("-r", "--src_ply", default="../src.ply", type=str, help="Path to source point cloud file.")
parse.add_argument("-t", "--tgt_ply", default="../tgt.ply", type=str, help="Path to target point cloud file.")
parse.add_argument("-d", "--data_path", default="../../mvs_data", type=str, help="Path to the DTU evaluation data.")
parse.add_argument("-o", "--output_path", default="./evaluation", type=str, help="Output path where all metrics and results will be stored.")
parse.add_argument("-l", "--light_setting", default="l3", type=str, help="DTU light setting.")
parse.add_argument("-p", "--representation", default="Points", type=str, help="Data representation (Points/Surface).")
parse.add_argument("-s", "--scene", default="1", type=str, help="Scene name or number (e.x. Ignatius, Truck, Barn ...or... 1, 4, 9).")
parse.add_argument("-v", "--voxel_size", default=0.2, type=float, help="Voxel size used for consistent downsampling.")
parse.add_argument("-x", "--max_dist", default=0.4, type=float, help="Max distance threshold for point matching.")
parse.add_argument("-e", "--data_set", default="none", type=str, help="Data set if a dataset-specific point-cloud comparison is to be made.")

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

def compare_point_clouds(src_ply, tgt_ply, max_dist, src_filt, tgt_filt):
    # compute bi-directional distance between point clouds
    md = 20

    dists_src = np.asarray(src_ply.compute_point_cloud_distance(tgt_ply))
    valid_inds_src = set(np.where(src_filt == 1)[0])
    valid_dists = set(np.where(dists_src <= md)[0])
    valid_inds_src.intersection_update(valid_dists)
    valid_inds_src = np.asarray(list(valid_inds_src))
    dists_src = dists_src[valid_inds_src]

    dists_tgt = np.asarray(tgt_ply.compute_point_cloud_distance(src_ply))
    valid_inds_tgt = set(np.where(tgt_filt == 1)[0])
    valid_dists = set(np.where(dists_tgt <= md)[0])
    valid_inds_tgt.intersection_update(valid_dists)
    valid_inds_tgt = np.asarray(list(valid_inds_tgt))
    dists_tgt = dists_tgt[valid_inds_tgt]

    # compute accuracy and competeness
    acc = np.mean(dists_src)
    comp = np.mean(dists_tgt)

    # measure incremental precision and recall values with thesholds from (0, 10*max_dist)
    th_vals = np.linspace(0, 3*max_dist, num=50)
    prec_vals = [ (len(np.where(dists_src <= th)[0]) / len(dists_src)) for th in th_vals ]
    rec_vals = [ (len(np.where(dists_tgt <= th)[0]) / len(dists_tgt)) for th in th_vals ]

    # compute precision and recall for given distance threshold
    prec = len(np.where(dists_src <= max_dist)[0]) / len(dists_src)
    rec = len(np.where(dists_tgt <= max_dist)[0]) / len(dists_tgt)

    # color point cloud for precision
    valid_src_ply = src_ply.select_by_index(valid_inds_src)
    src_size = len(valid_src_ply.points)
    cmap = plt.get_cmap("hot_r")
    colors = cmap(np.minimum(dists_src, max_dist) / max_dist)[:, :3]
    valid_src_ply.colors = o3d.utility.Vector3dVector(colors)

    # color invalid points precision
    invalid_src_ply = src_ply.select_by_index(valid_inds_src, invert=True)
    cmap = plt.get_cmap("winter")
    colors = cmap(np.ones(len(invalid_src_ply.points)))[:, :3]
    invalid_src_ply.colors = o3d.utility.Vector3dVector(colors)

    # color point cloud for recall
    valid_tgt_ply = tgt_ply.select_by_index(valid_inds_tgt)
    tgt_size = len(valid_tgt_ply.points)
    cmap = plt.get_cmap("hot_r")
    colors = cmap(np.minimum(dists_tgt, max_dist) / max_dist)[:, :3]
    valid_tgt_ply.colors = o3d.utility.Vector3dVector(colors)

    # color invalid points recall
    invalid_tgt_ply = tgt_ply.select_by_index(valid_inds_tgt, invert=True)
    cmap = plt.get_cmap("winter")
    colors = cmap(np.ones(len(invalid_tgt_ply.points)))[:, :3]
    invalid_tgt_ply.colors = o3d.utility.Vector3dVector(colors)

    return (valid_src_ply + invalid_src_ply, valid_tgt_ply + invalid_tgt_ply), (acc,comp), (prec, rec), (th_vals, prec_vals, rec_vals), (src_size, tgt_size)

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
    data_path = ARGS.data_path
    method = ARGS.method
    scan_num = ARGS.scene
    output_path = ARGS.output_path
    voxel_size = ARGS.voxel_size
    max_dist = ARGS.max_dist
    data_set = ARGS.data_set

    output_path = os.path.join(output_path, "{}_{}".format(method, str(scan_num).zfill(3)))

    # create output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)



    ##### Load in point clouds #####
    print("Loading point clouds...")
    src_ply = read_point_cloud(src_path, voxel_size)
    tgt_ply = read_point_cloud(tgt_path, voxel_size)



    ##### Create masks #####
    # read in matlab bounding box, mask, and resolution
    mask_filename = "ObsMask{}_10.mat".format(scan_num)
    mask_path = os.path.join(data_path, "ObsMask", mask_filename)
    data = sio.loadmat(mask_path)
    bounds = np.asarray(data["BB"])
    min_bound = bounds[0,:]
    max_bound = bounds[1,:]
    mask = np.asarray(data["ObsMask"])
    res = int(data["Res"])

    # read in matlab gt plane 
    mask_filename = "Plane{}.mat".format(scan_num)
    mask_path = os.path.join(ARGS.data_path, "ObsMask", mask_filename)
    data = sio.loadmat(mask_path)
    P = np.asarray(data["P"])

    # build points filter based on input mask
    src_filt = build_src_points_filter(src_ply, min_bound, res, mask)

    # build points filter based on input mask
    tgt_filt = build_tgt_points_filter(tgt_ply, P)



    ##### Compute metrics between point clouds #####
    print("Computing metrics between point clouds...")
    (precision_ply, recall_ply), (acc,comp), (prec, rec), (th_vals, prec_vals, rec_vals), (src_size, tgt_size) \
            = compare_point_clouds(src_ply, tgt_ply, max_dist, src_filt, tgt_filt)



    ##### Save metrics #####
    print("Saving evaluation statistics...")
    # save precision point cloud
    precision_path = os.path.join(output_path, "precision_{}.ply".format(method))
    save_ply(precision_path, precision_ply)

    # save recall point cloud
    recall_path = os.path.join(output_path, "recall_{}.ply".format(method))
    save_ply(recall_path, recall_ply)

    # create plots for incremental threshold values
    plot_filename = os.path.join(output_path, "metrics_{}.png".format(method))
    plt.plot(th_vals, prec_vals, th_vals, rec_vals)
    plt.title("Precision and Recall (t={}mm)".format(max_dist))
    plt.xlabel("threshold")
    plt.vlines(max_dist, 0, 1, linestyles='dashed', label='t')
    plt.legend(("precision", "recall"))
    plt.grid()
    plt.savefig(plot_filename)

    # write all metrics to the evaluation file
    stats_file = os.path.join(output_path, "evaluation_metrics_{}.txt".format(method))
    with open(stats_file, 'w') as f:
        f.write("Method: {}\n".format(method))
        f.write("Voxel_size: {:0.3f}mm | Distance threshold: {:0.3f}mm\n".format(voxel_size, max_dist))
        f.write("Source point cloud size: {}\n".format(src_size))
        f.write("Target point cloud size: {}\n".format(tgt_size))
        f.write("Accuracy: {:0.3f}mm\n".format(acc))
        f.write("Completness: {:0.3f}mm\n".format(comp))
        f.write("Precision: {:0.3f}\n".format(prec))
        f.write("Recall: {:0.3f}\n".format(rec))


if __name__=="__main__":
    main()
