import sys, os
import numpy as np
import matplotlib.pyplot as plt

from cvt.io import read_pfm

def sparse_dense_eval(sparse_depth, dense_depth, i, est_depth=None):
    assert(sparse_depth.shape == dense_depth.shape)

    valid_sparse = np.where(sparse_depth > 0, 1, 0)
    valid_dense = np.where(dense_depth > 0, 1, 0) * valid_sparse

    dense_ae = np.abs(sparse_depth - dense_depth)
    dense_ae *= valid_dense

    # Visualize
    #plt.imshow(dense_ae, cmap="hot", vmin=0, vmax=0.3)
    #plt.savefig(f"err_{i:04d}.png")
    #plt.clf()
    #plt.close()

    dense_mae = dense_ae.sum() / (valid_dense.sum()+1e-5)

    if est_depth != None:
        assert(sparse_depth.shape == est_depth.shape)
        est_ae = np.abs(sparse_depth - est_depth)
        est_ae *= valid_sparse
        est_mae = dense_ae.sum() / (valid_sparse.sum()+1e-5)

        return dense_mae, est_mae
    else:
        return dense_mae


def main():
    sparse_depth_path = sys.argv[1]
    dense_depth_path = sys.argv[2]
    if len(sys.argv) == 4:
        est_depth_path = sys.argv[3]

    sparse_files = os.listdir(sparse_depth_path)
    sparse_files = [os.path.join(sparse_depth_path,s) for s in sparse_files if s[-3:] == "pfm" ]
    sparse_files.sort()

    dense_files = os.listdir(dense_depth_path)
    dense_files = [os.path.join(dense_depth_path,s) for s in dense_files if s[-3:] == "pfm" ]
    dense_files.sort()
    assert(len(sparse_files) == len(dense_files))
    dense_mae = []

    if len(sys.argv) == 4:
        est_files = os.listdir(est_depth_path)
        est_files = [s for s in est_files if s[-3:] == "pfm" ]
        est_files.sort()
        assert(len(sparse_files) == len(est_files))
        est_mae = []


    for i in range(len(sparse_files)):
        sparse_depth = read_pfm(sparse_files[i])
        dense_depth = read_pfm(dense_files[i])

        if len(sys.argv) == 4:
            est_depth = read_pfm(est_files[i])
            dmae, emae = sparse_dense_eval(sparse_depth, dense_depth, i, est_depth)
            dense_mae.append(dmae)
            est_mae.append(emae)
        else:
            dmae = sparse_dense_eval(sparse_depth, dense_depth, i)
            dense_mae.append(dmae)

    dense_mae = np.asarray(dense_mae)
    print(f"Dense MAE: {dense_mae.mean()}")

if __name__=="__main__":
    main()
