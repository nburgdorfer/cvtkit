#!/bin/bash

SCENES=(Barn Caterpillar Church Courthouse Ignatius Meetingroom Truck)

for SCENE in ${SCENES[@]}
do
	echo "Working on ${SCENE}"
	python convert_from_log.py --log_file /media/nate/Data/TNT/training/Cameras/${SCENE}/camera_pose.log --output_path /media/nate/Data/TNT/training/Cameras/${SCENE}/ --old_cam_path /media/nate/Data/TNT/training/Cameras_old/${SCENE}/
done
