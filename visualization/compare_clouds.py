import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys
import os


def read_point_cloud(ply_path, size=0.1):
    if(ply_path[-3:] != "ply"):
        print("{} is not a '.ply' file.".format(ply_path))
        sys.exit()

    ply = o3d.io.read_point_cloud(ply_path)
    ply = ply.voxel_down_sample(voxel_size=size)

    return ply

def compare_point_clouds(ply1, ply2, max_dist=1.0):
    # compute bi-directional distance between point clouds
    dists_12 = np.asarray(ply1.compute_point_cloud_distance(ply2))
    dists_21 = np.asarray(ply2.compute_point_cloud_distance(ply1))

    # compute accuracy and competeness
    acc = np.mean(dists_12)
    comp = np.mean(dists_21)

    # measure incremental precision and recall values with thesholds from (0, 10*max_dist)
    th_vals = np.linspace(0, 3*max_dist, num=50)
    prec_vals = [ (len(np.where(dists_12 <= th)[0]) / len(dists_12)) for th in th_vals ]
    rec_vals = [ (len(np.where(dists_21 <= th)[0]) / len(dists_21)) for th in th_vals ]

    # compute precision and recall for given distance threshold
    prec = len(np.where(dists_12 <= max_dist)[0]) / len(dists_12)
    rec = len(np.where(dists_21 <= max_dist)[0]) / len(dists_21)

    # color point cloud for precision
    cmap = plt.get_cmap("hot_r")
    colors = cmap(np.minimum(dists_12, max_dist) / max_dist)[:, :3]
    ply1.colors = o3d.utility.Vector3dVector(colors)

    # color point cloud for recall
    cmap = plt.get_cmap("hot_r")
    colors = cmap(np.minimum(dists_21, max_dist) / max_dist)[:, :3]
    ply2.colors = o3d.utility.Vector3dVector(colors)

    return (ply1, ply2), (acc,comp), (prec, rec), (th_vals, prec_vals, rec_vals)

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
    if(len(sys.argv) != 6):
        print("Error: usage python3 {} <source-point-cloud> <target-point-cloud> <output-path> <voxel_size> <max-distance>".format(sys.argv[0]))
        sys.exit()

    src_path = sys.argv[1]
    tgt_path = sys.argv[2]
    output_path = sys.argv[3]
    voxel_size = float(sys.argv[4])
    max_dist = float(sys.argv[5])

    print("Loading point clouds...")
    src_ply = read_point_cloud(src_path, voxel_size)
    tgt_ply = read_point_cloud(tgt_path, voxel_size)

    src_size = len(src_ply.points)
    tgt_size = len(tgt_ply.points)

    print("Computing metrics between point clouds...")
    (precision_ply, recall_ply), (acc,comp), (prec, rec), (th_vals, prec_vals, rec_vals) = compare_point_clouds(src_ply, tgt_ply, max_dist)

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
