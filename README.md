# Computer Vision Toolkit (CVT)
A collection of some useful (or maybe just interesting) computer vision tools.

### Visualizing Epipolar Geometry
The ```pyfmatrix_from_P_Viewer.py``` and ```pyfmatrix_viewer.py``` scripts can be used to view the projection of the epipolar line from a pixel in one image to a corresponding image. Example data is provided in the ```data/examples/``` directory:

Here is an example of the use of the ```pyfmatrix_from_P_viewer.py``` script:
```
> python3 pyfmatrix_from_P_viewer.py ../data/examples/000_img.jpg ../data/examples/001_img.jpg ../data/examples/K.txt ../data/examples/000_P.txt ../data/examples/001_P.txt
```

This script takes as input two images, the intrinsics matrix of the camera, and the extrinsics of each view.
