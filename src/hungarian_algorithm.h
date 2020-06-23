#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <string>
#define FLT_EPSILON 0.1

std::vector<std::pair<float, Eigen::Vector2i>> HungarianAlgorithm(const Eigen::MatrixXf& _hungarianMat);
