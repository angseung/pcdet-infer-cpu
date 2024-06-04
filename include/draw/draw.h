#ifndef __DRAW_H__
#define __DRAW_H__

#include <cstddef>
#include <opencv2/opencv.hpp>

#include "type.h"

cv::Mat drawBirdsEyeView(const size_t &points_size, const float *points_data,
                         const std::vector<vueron::BndBox> &boxes,
                         const std::vector<float> &scores,
                         const std::vector<size_t> &labels);

#endif  // __DRAW_H__
