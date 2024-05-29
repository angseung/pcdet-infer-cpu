#ifndef DRAW_H
#define DRAW_H

#include <cstddef>
#include <opencv2/opencv.hpp>

cv::Mat drawBirdsEyeView(size_t points_size, const float *points_data,
                         const std::vector<vueron::BndBox> &boxes,
                         const std::vector<float> &scores,
                         const std::vector<size_t> &labels);

#endif
