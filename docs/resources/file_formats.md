## SFM Format
Below is an example of camera parameters encoded in *SFM Format*.

The first line is a tag that identifies the following information is the homogeneous transformation (or extrinsics) matrix of the camera. Rows of each matrix in this format are encoded on separate lines, with a space between each matrix element.

Following the extrinsics matrix is a blank line, followed by the intrinsics tag. The intrinsics matrix follows the same format as the extrinsics matrix above.

The last line is optional and represents some usefull scene information, including the depth bounds for the view, as well as expected (or chosen) plane hypothesis information. The first element (**0.533**) and last element (**2.609**) are the minimum and maximum depth bounds, respectively. The second element is the expected plane spacing resolution and the third element is the number of plane hypothesis to be used. These values are typically used in *Multi-View Stereo (MVS)* applications, or any derivative/adaptation of the plane-sweeping aglorithm.

```
extrinsic
-0.946 -0.019 0.322 0.558 
0.031 -0.998 0.032 0.052 
0.321 0.040 0.946 1.947 
0.000 0.000 0.000 1.000

intrinsic
1165.100 0.000 963.452 
0.000 1139.208 530.837 
0.000 0.000 1.000

0.533 0.008 256 2.609
```

It is also supported to use different subsets of these elements. The possible subsets include:

- 	```
	min_depth_bound plane_interval
	```

-	```
	min_depth_bound plane_interval depth_planes
	```

with the additional elements either being inferred or using default values.

Datasets utilizing the above format:
- DTU
- Tanks & Temples
- BlendedMVS

## Trajectory Format
Below is an example of camera parameters encoded into a *Trajectory (.log) File Format*.

Every five lines is a camera entry, with no blank lines spearating the entries.

The first line of every entry encodes metadata, usually including usefull information, such as the view (or frame) number. This line is a three-number metedata tag, with the view number typically encoded in the third slot. The **Tanks & Temples** dataset, however, encodes the view number in the first (and second) metadata slot(s). Since this library mainly works with Trajectory format files from this dataset, we will use this dataset-specific encoding.

The following four lines encode the homogeneous transformation (or extrinsics) matrix.

```
0 0 0
6.375 0.071 2.49 -16.267
-2.499 0.308 6.368 -10.192
-0.046 -6.841 0.313 -22.115
0.000 0.000 0.000 1.000
1 1 0
6.375 0.085 2.500 -16.630
-2.501 0.306 6.367 -10.067
-0.032 -6.841 0.316 -22.117
0.000 0.000 0.000 1.000
2 2 0
6.374 0.097 2.500 -17.020
-2.502 0.304 6.367 -9.895
-0.020 -6.840 0.319 -22.070
0.000 0.000 0.000 1.000
3 3 0
6.348 0.118 2.565 -17.369
-2.568 0.319 6.340 -9.799
-0.010 -6.839 0.340 -22.033
0.000 0.000 0.000 1.000
```

The advantage of using the Trajectory file format is that only a single file is needed to encode the camera trajectories, aiding in dataset organization. The camera intrinsics, however, will need to be inferred or provided separately.

Datasets utilizing the above format:
- DTU
- Tanks & Temples
- BlendedMVS

## TUM Format
Below is an example of camera parameters encoded in *TUM Format*.

Each line represents a single camera entry, encoded as follows:

- `timestamp tx ty tz qx qy qz qw`

The first element in every line is the timestamp corresponding to the time-of-capture for the camera entry.

The next three elements encode the position of the camera optical center.

The last four elements encode the orientation of the camera optical axis in unit quaternion form.


```
1638570918.135257600 -2.927 -3.658 -10.617 0.014 -0.584 0.142 0.798
1638570918.202363648 -2.934 -3.665 -10.622 0.021 -0.585 0.144 0.794
1638570918.269449728 -2.942 -3.671 -10.628 0.027 -0.586 0.146 0.795
```

Much like Trajectory format, the advantage of using the TUM file format is that only a single file is needed to encode the camera trajectories. The camera intrinsics must again be inferred or provided separately.

TUM format is typically used for real-time applications, SLAM-based algorithms, ROS-collected datasets, etc.
