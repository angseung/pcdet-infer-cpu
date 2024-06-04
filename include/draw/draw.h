#ifndef __DRAW_H__
#define __DRAW_H__

#include <cstddef>
#include <opencv2/opencv.hpp>

#include "type.h"

inline cv::Point2f rotatePoint(const cv::Point2f &point, const float &angle) {
  float x = point.x * std::cos(angle) - point.y * std::sin(angle);
  float y = point.x * std::sin(angle) + point.y * std::cos(angle);
  return cv::Point2f(x, y);
}

cv::Mat drawBirdsEyeView(const size_t &points_size, const float *points_data,
                         const std::vector<vueron::BndBox> &boxes,
                         const std::vector<float> &scores,
                         const std::vector<size_t> &labels);

#endif  // __DRAW_H__
