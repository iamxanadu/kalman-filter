cmake_minimum_required(VERSION 3.15)
project (LinearKalmanFilter)

set(SOURCE src/kalman_filter.cpp)
add_library(liblinkalman ${SOURCE})