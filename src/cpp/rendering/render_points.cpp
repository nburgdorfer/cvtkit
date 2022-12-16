// PyBind11 Includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

// OpenCV Includes
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// OpenMP Includes
#include <omp.h>

// C++ Includes
#include <stdlib.h>
#include <vector>

// Eigen Includes
#include <Eigen/Dense>

namespace py = pybind11;

using namespace std;
using namespace cv;
using namespace pybind11::literals;

using Eigen::MatrixXf;

void display_depth(const Mat map, string filename) {
    Size size = map.size();
    // crop 20 pixels
    Mat cropped = map(Rect(14,14,size.width-30,size.height-30));

    int min = 425;
    int max = 937;

    cropped = (cropped-min) * 255 / (max-min);
    Mat output;
    threshold(cropped, output,0, 255, THRESH_TOZERO);
    imwrite(filename, output);
}

float sigmoid(float x, float scale) {
	float input = (scale*x);
	return 1 / (1 + exp(-input));
}


MatrixXf render(const vector<int> &shape, const vector<vector<float>> points, const vector<float> values, const vector<float> &cam) {
	// grab depth map shape
	int rows = shape[0];
	int cols = shape[1];

	int num_points = points.size();

	// return container initialization
	MatrixXf rendered_map = MatrixXf::Zero(rows, cols);

	int cam_shape[3] = {2,4,4};

	Mat P = Mat::zeros(4,4,CV_32F);
	Mat K = Mat::zeros(4,4,CV_32F);

#pragma omp parallel num_threads(12)
{
	#pragma omp for collapse(3)
	for (int i=0; i<cam_shape[0]; ++i) {
		for (int j=0; j<cam_shape[1]; ++j) {
			for (int k=0; k<cam_shape[2]; ++k) {
				int ind = (i*cam_shape[1]*cam_shape[2]) + (j*cam_shape[2]) + (k);
				if(i==0){
					P.at<float>(j,k) = cam[ind];
				} else if(i==1) {
					K.at<float>(j,k) = cam[ind];
				}
			}
		}
	}
} //omp parallel

	// correct the last row of the intrinsics
	K.at<float>(3,0) = 0;
	K.at<float>(3,1) = 0;
	K.at<float>(3,2) = 0;
	K.at<float>(3,3) = 1;

	// compute the forwards projection
	Mat f_proj =  K * P;

#pragma omp parallel num_threads(12)
{
    #pragma omp for
	for (int ind=0; ind<num_points; ++ind) {
		Mat x_3d = Mat::ones(4,1,CV_32F);
		x_3d.at<float>(0,0) = points[ind][0];
		x_3d.at<float>(0,1) = points[ind][1];
		x_3d.at<float>(0,2) = points[ind][2];

		// calculate pixel location in target image
		Mat x_2 = f_proj * x_3d;

		x_2.at<float>(0,0) = x_2.at<float>(0,0)/x_2.at<float>(2,0);
		x_2.at<float>(1,0) = x_2.at<float>(1,0)/x_2.at<float>(2,0);
		x_2.at<float>(2,0) = x_2.at<float>(2,0)/x_2.at<float>(2,0);

		// take the floor to get the row and column pixel locations
		int c_p = (int) floor(x_2.at<float>(0,0));
		int r_p = (int) floor(x_2.at<float>(1,0));
		
		// ignore if pixel projection falls outside the image
		if (c_p < 0 || c_p >= cols || r_p < 0 || r_p >= rows) {
			continue;
		}

		// keep higher value
		float curr_val = values[ind];
		if (rendered_map(r_p, c_p) <= 0.0) {
			rendered_map(r_p, c_p) = curr_val;
		} else {
			rendered_map(r_p, c_p) = std::min(rendered_map(r_p, c_p), curr_val);
		}
	}

} //omp parallel

	return rendered_map;
}

PYBIND11_MODULE(render_points, m) {
	m.doc() = "MVS Utilities C++ Pluggin";
	m.def(	"render",
			&render,
			"A function which renders a set of 3D points into the image plane for a given view using a set of point values",
			"shape"_a,
			"points"_a,
			"values"_a,
			"cam"_a);
}
