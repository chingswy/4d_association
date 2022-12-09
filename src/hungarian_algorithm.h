#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <string>
#include <float.h>

std::vector<std::pair<float, Eigen::Vector2i>> HungarianAlgorithm(const Eigen::MatrixXf& _hungarianMat);
