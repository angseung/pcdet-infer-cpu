#ifndef __DRAW_H__
#define __DRAW_H__

#include <cstddef>
#include <opencv2/opencv.hpp>

#include "type.h"

inline cv::Point2f rotatePoint(const cv::Point2f &point, const float angle) {
  const float x = point.x * std::cos(angle) - point.y * std::sin(angle);
  const float y = point.x * std::sin(angle) + point.y * std::cos(angle);
  return cv::Point2f{x, y};
}

void drawBirdsEyeView(size_t point_buf_len, size_t point_stride,
                      const float *points_data,
                      const std::vector<vueron::BndBox> &boxes,
                      const std::vector<float> &scores,
                      const std::vector<size_t> &labels, float scale,
                      cv::Mat &image);

#endif  // __DRAW_H__
