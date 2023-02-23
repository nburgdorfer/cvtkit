#!/bin/bash

FUSION_DIR=/media/nate/Data/DTU/
EVAL_DIR=/media/nate/Data/Evaluation/dtu/mvs_data/Points/stl/
#SCANS=({1..24} {28..53} {55..72} {74..77} {82..128})
SCANS=(1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118)

for SCAN in ${SCANS[@]}
do
    echo "Generating ground-truth visibility maps for scan ${SCAN}"

    # pad scan number
    printf -v PADDED_SCAN_NUM "%03d" $SCAN
    # creat scan name
    SCAN_DIR="scan${PADDED_SCAN_NUM}/"

	python generate_dense_depths.py \
		--gt_depth_dir ${FUSION_DIR}GT_Depths/${SCAN_DIR}/ \
		--camera_dir ${FUSION_DIR}Cameras/ \
		--output_dir ${EVAL_DIR} \
		--scan ${SCAN} \
		--voxel_size 0.03
done
