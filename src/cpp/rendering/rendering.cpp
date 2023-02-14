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


vector<float> render_to_tgt(const vector<int> &vol_shape, const vector<float> depth_values, const vector<float> original_depth, const vector<float> original_conf, const vector<float> &reference_cam, const vector<float> &target_cam, const float &scale) {
	// grab shape of volume
	int depth_planes = vol_shape[0];
	int rows = vol_shape[1];
	int cols = vol_shape[2];

	// return container initialization
	vector<float> rendered_volume(depth_planes*rows*cols, 0);

	int cam_shape[3] = {2,4,4};

	//Mat depth_slice = Mat::zeros(rows,cols,CV_32F);

	Mat P_ref = Mat::zeros(4,4,CV_32F);
	Mat K_ref = Mat::zeros(4,4,CV_32F);
	Mat P_tgt = Mat::zeros(4,4,CV_32F);
	Mat K_tgt = Mat::zeros(4,4,CV_32F);

#pragma omp parallel num_threads(12)
{
	#pragma omp for collapse(3)
	for (int i=0; i<cam_shape[0]; ++i) {
		for (int j=0; j<cam_shape[1]; ++j) {
			for (int k=0; k<cam_shape[2]; ++k) {
				int ind = (i*cam_shape[1]*cam_shape[2]) + (j*cam_shape[2]) + (k);
				if(i==0){
					P_ref.at<float>(j,k) = reference_cam[ind];
					P_tgt.at<float>(j,k) = target_cam[ind];
				} else if(i==1) {
					K_ref.at<float>(j,k) = reference_cam[ind];
					K_tgt.at<float>(j,k) = target_cam[ind];
				}
			}
		}
	}

	// correct the last row of the intrinsics
	K_ref.at<float>(3,0) = 0;
	K_ref.at<float>(3,1) = 0;
	K_ref.at<float>(3,2) = 0;
	K_ref.at<float>(3,3) = 1;

	K_tgt.at<float>(3,0) = 0;
	K_tgt.at<float>(3,1) = 0;
	K_tgt.at<float>(3,2) = 0;
	K_tgt.at<float>(3,3) = 1;

} //omp parallel

	// compute the rotation, translation, and camera centers for the target view
	Mat R_tgt = P_tgt(Rect(0,0,3,3));
	Mat z_tgt = R_tgt(Rect(0,2,3,1));
	Mat t_tgt = P_tgt(Rect(3,0,1,3));
	Mat C_tgt = -R_tgt.t()*t_tgt;

	// compute the backwards and forwards projections
	Mat b_proj =  P_ref.inv() * K_ref.inv();
	Mat f_proj =  K_tgt * P_tgt;

#pragma omp parallel num_threads(12)
{
    #pragma omp for collapse(3)
	for (int d=0; d<depth_planes; ++d) {
		for (int r=0; r<rows; ++r) {
			for (int c=0; c<cols; ++c) {
				float depth = depth_values[d];

				// compute 3D world coord of back projection
				Mat x_1(4,1,CV_32F);
				x_1.at<float>(0,0) = depth * c;
				x_1.at<float>(1,0) = depth * r;
				x_1.at<float>(2,0) = depth;
				x_1.at<float>(3,0) = 1;

				Mat X_world = b_proj * x_1;
				X_world.at<float>(0,0) = X_world.at<float>(0,0) / X_world.at<float>(0,3);
				X_world.at<float>(0,1) = X_world.at<float>(0,1) / X_world.at<float>(0,3);
				X_world.at<float>(0,2) = X_world.at<float>(0,2) / X_world.at<float>(0,3);

				// calculate pixel location in target image
				Mat x_2 = f_proj * X_world;

				x_2.at<float>(0,0) = x_2.at<float>(0,0)/x_2.at<float>(2,0);
				x_2.at<float>(1,0) = x_2.at<float>(1,0)/x_2.at<float>(2,0);

				// take the floor to get the row and column pixel locations
				int c_p = (int) floor(x_2.at<float>(0,0));
				int r_p = (int) floor(x_2.at<float>(1,0));
				
				// ignore if pixel projection falls outside the image
				if (c_p < 0 || c_p >= cols || r_p < 0 || r_p >= rows) {
					continue;
				}

				// calculate the projection depth from reference image plane
				Mat diff = Mat::zeros(3,1,CV_32F);
				diff.at<float>(0,0) = X_world.at<float>(0,0) - C_tgt.at<float>(0,0);
				diff.at<float>(0,1) = X_world.at<float>(0,1) - C_tgt.at<float>(0,1);
				diff.at<float>(0,2) = X_world.at<float>(0,2) - C_tgt.at<float>(0,2);

				//project on z-axis of target cam
				Mat projection = z_tgt * diff;
				float proj_depth = projection.at<float>(0);

				// calculate the current index
				int ind = (d*rows*cols) + (r*cols) + c;
				int proj_ind = (r_p*cols) + c_p;

				float depth_diff = original_depth[proj_ind] - proj_depth;
				float sig_output = sigmoid(depth_diff, scale);
				if(d==54 && c==120 && r==120) {
					cout << proj_depth << endl;
					cout << original_depth[proj_ind] << endl;
					cout << depth_diff << endl;
					cout << sig_output << endl;
					exit(0);
				}

				rendered_volume[ind] = original_conf[proj_ind] * sig_output;
			}
		}

	}
} //omp parallel

	return rendered_volume;
}

vector<float> render_to_ref(const vector<int> &shape, const vector<float> depth_map, const vector<float> conf_map, const vector<float> &reference_cam, const vector<float> &target_cam) {
	// grab depth map shape
	int rows = shape[0];
	int cols = shape[1];

	// return container initialization
	vector<float> rendered_map(2*rows*cols, 0);

	int cam_shape[3] = {2,4,4};

	Mat P_ref = Mat::zeros(4,4,CV_32F);
	Mat K_ref = Mat::zeros(4,4,CV_32F);
	Mat P_tgt = Mat::zeros(4,4,CV_32F);
	Mat K_tgt = Mat::zeros(4,4,CV_32F);

#pragma omp parallel num_threads(12)
{
	#pragma omp for collapse(3)
	for (int i=0; i<cam_shape[0]; ++i) {
		for (int j=0; j<cam_shape[1]; ++j) {
			for (int k=0; k<cam_shape[2]; ++k) {
				int ind = (i*cam_shape[1]*cam_shape[2]) + (j*cam_shape[2]) + (k);
				if(i==0){
					P_ref.at<float>(j,k) = reference_cam[ind];
					P_tgt.at<float>(j,k) = target_cam[ind];
				} else if(i==1) {
					K_ref.at<float>(j,k) = reference_cam[ind];
					K_tgt.at<float>(j,k) = target_cam[ind];
				}
			}
		}
	}

	// correct the last row of the intrinsics
	K_ref.at<float>(3,0) = 0;
	K_ref.at<float>(3,1) = 0;
	K_ref.at<float>(3,2) = 0;
	K_ref.at<float>(3,3) = 1;

	K_tgt.at<float>(3,0) = 0;
	K_tgt.at<float>(3,1) = 0;
	K_tgt.at<float>(3,2) = 0;
	K_tgt.at<float>(3,3) = 1;

} //omp parallel

	// compute the rotation, translation, and camera centers for the target view
	Mat R_ref = P_ref(Rect(0,0,3,3));
	Mat z_ref = R_ref(Rect(0,2,3,1));
	Mat t_ref = P_ref(Rect(3,0,1,3));
	Mat C_ref = -R_ref.t()*t_ref;

	// compute the backwards and forwards projections
	Mat b_proj =  P_tgt.inv() * K_tgt.inv();
	Mat f_proj =  K_ref * P_ref;

#pragma omp parallel num_threads(12)
{
    #pragma omp for collapse(2)
	for (int r=0; r<rows; ++r) {
		for (int c=0; c<cols; ++c) {
			// calculate the current index
			int ind = (r*cols) + c;

			float depth = depth_map[ind];

			// compute 3D world coord of back projection
			Mat x_1(4,1,CV_32F);
			x_1.at<float>(0,0) = depth * c;
			x_1.at<float>(1,0) = depth * r;
			x_1.at<float>(2,0) = depth;
			x_1.at<float>(3,0) = 1;

			Mat X_world = b_proj * x_1;
			X_world.at<float>(0,0) = X_world.at<float>(0,0) / X_world.at<float>(0,3);
			X_world.at<float>(0,1) = X_world.at<float>(0,1) / X_world.at<float>(0,3);
			X_world.at<float>(0,2) = X_world.at<float>(0,2) / X_world.at<float>(0,3);
			X_world.at<float>(0,3) = X_world.at<float>(0,3) / X_world.at<float>(0,3);

			// calculate pixel location in target image
			Mat x_2 = f_proj * X_world;

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

			// calculate the projection depth from reference image plane
			Mat diff = Mat::zeros(3,1,CV_32F);
			diff.at<float>(0,0) = X_world.at<float>(0,0) - C_ref.at<float>(0,0);
			diff.at<float>(0,1) = X_world.at<float>(0,1) - C_ref.at<float>(0,1);
			diff.at<float>(0,2) = X_world.at<float>(0,2) - C_ref.at<float>(0,2);

			//project on z-axis of target cam
			Mat projection = z_ref * diff;
			float proj_depth = projection.at<float>(0);

			// compute projection index
			int proj_ind = (r_p*cols) + c_p;

			/* 
			 * Keep the closer (smaller) projection depth.
			 * A previous projection could have already populated the current pixel.
			 * If it is 0, no previous projection to this pixel was seen.
			 * Otherwise, we need to overwrite only if the current estimate is closer (smaller value).
			 */
			if (rendered_map[proj_ind] > 0) {
				if(rendered_map[proj_ind] > proj_depth) {
					rendered_map[proj_ind] = proj_depth;
					rendered_map[proj_ind+(rows*cols)] = conf_map[ind];
				}
			} else {
				rendered_map[proj_ind] = proj_depth;
				rendered_map[proj_ind+(rows*cols)] = conf_map[ind];
			}
		}
	}

} //omp parallel

	return rendered_map;
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

PYBIND11_MODULE(rendering, m) {
	m.doc() = "MVS Utilities C++ Pluggin";
	m.def(	"render",
			&render,
			"A function which renders a set of 3D points into the image plane for a given view using a set of point values",
			"shape"_a,
			"points"_a,
			"values"_a,
			"cam"_a);
	m.def(	"render_to_tgt",
			&render_to_tgt,
			"A function which renders a reference volume onto a target volume",
			"vol_shape"_a,
			"depth_values"_a,
			"original_depth"_a,
			"original_conf"_a,
			"reference_cam"_a,
			"target_cam"_a,
			"scale"_a);
	m.def(	"render_to_ref",
			&render_to_ref,
			"A function which renders a target depth map into a reference view",
			"shape"_a,
			"depth_map"_a,
			"conf_map"_a,
			"reference_cam"_a,
			"target_cam"_a);
}
