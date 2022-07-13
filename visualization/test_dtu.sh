#!/bin/bash

DATA_PATH=/media/nate/Data/Evaluation/dtu/mvs_data/
OUTPUT_PATH=./evaluation

SCANS=(1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118)

for SCAN in ${SCANS[@]}
do
	echo -e "\e[1;33mComparing point-clouds for scan ${SCAN}\e[0;37m"

	printf -v PADDED_SCAN_NUM "%03d" $SCAN
	TGT_PLY=/media/nate/Data/Evaluation/dtu/mvs_data/Points/stl/stl${PADDED_SCAN_NUM}_total.ply
	METHOD=fusion
	SRC_PLY=/media/nate/Data/DMFNet/dtu/V7-0-0_DTU/scan${PADDED_SCAN_NUM}/point_clouds/merged.ply

	python compare_clouds.py \
		--method $METHOD \
		--scene $SCAN \
		--data_path $DATA_PATH \
		--output_path $OUTPUT_PATH \
		--src_ply $SRC_PLY \
		--tgt_ply $TGT_PLY



	METHOD=gipuma
	SRC_PLY=/media/nate/Data/UCSNet/Points/dtu/ucsnet${PADDED_SCAN_NUM}_l3.ply

	python compare_clouds.py \
		--method $METHOD \
		--scene $SCAN \
		--data_path $DATA_PATH \
		--output_path $OUTPUT_PATH \
		--src_ply $SRC_PLY \
		--tgt_ply $TGT_PLY
done
