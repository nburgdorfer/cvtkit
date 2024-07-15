import sys, os
import numpy as np




def load_points(points_file, output_file, error_th=1.0, track_len_th=4):
    points = []
    with open(points_file, 'r') as pf:
        lines = pf.readlines()[3:]

        for i, l in enumerate(lines):
            l = l.strip().split()
            point = l[1:7]
            error = float(l[7])
            track = l[8:]
            track_len = len(track) // 2
            
            if error <= error_th and track_len >= track_len_th:
                points.append(np.asarray(point))

    # write header meta-data
    ply_str = ""
    ply_str += "ply\n"
    ply_str += "format ascii 1.0\n"
    ply_str += "comment Right-Handed System\n"
    ply_str += f"element vertex {len(points)}\n"
    ply_str += "property float x\n"
    ply_str += "property float y\n"
    ply_str += "property float z\n"
    ply_str += "property uchar red\n"
    ply_str += "property uchar green\n"
    ply_str += "property uchar blue\n"
    ply_str += "end_header\n"

    for point in points:
        ply_str += f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]} {point[5]}\n"

    with open(output_file, 'w') as of:
        of.write(ply_str)


def main():
    points_file = sys.argv[1]
    output_file = sys.argv[2]
    points = load_points(points_file, output_file)


if __name__=="__main__":
    main()
