cmake_minimum_required(VERSION 3.10)

project( render_points LANGUAGES CXX )

find_package( OpenCV REQUIRED )

find_package(OpenMP)

find_package(Eigen3)

find_package( pybind11 REQUIRED )
pybind11_add_module(render_points render_points.cpp)

target_link_libraries(render_points PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} Eigen3::Eigen)

