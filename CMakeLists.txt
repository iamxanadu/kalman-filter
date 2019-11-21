cmake_minimum_required(VERSION 3.15)
project (KalmanFilter)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(SOURCE src/kalman_filter.cpp)
include_directories(include/)

add_library(libkalman ${SOURCE})
target_link_libraries (libkalman Eigen3::Eigen)