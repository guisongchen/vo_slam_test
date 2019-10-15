#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

// std
#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <list>
#include <memory>
#include <set>

using namespace std;

// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::Quaterniond;

// for OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using cv::Mat;

// for Sophus
#include <sophus/se3.h>
#include <sophus/scso3.h>
#include <sophus/sim3.h>
using Sophus::SE3;

#endif
