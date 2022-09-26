#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

#include <iostream>
#include <memory>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <unordered_map>
#include <string>

using namespace std;

// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;
using Eigen::Vector3d;

// for Sophus
#include <sophus/se3.h>
using Sophus::SE3;

// for cv
#include <opencv2/core/core.hpp>
using cv::Mat;

#endif