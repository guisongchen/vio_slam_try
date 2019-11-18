#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

#include <vector>
#include <map>
#include <iostream>
#include <memory>
#include <string>
using std::cout;
using std::endl;

#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::MatrixXd;
using Eigen::VectorXd;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using cv::Mat;

#endif
