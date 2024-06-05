#include "draw/draw.h"

#include <cmath>
#include <vector>

#include "config.h"
#include "pcdet-infer-cpu/common/metadata.h"
#include "pcdet-infer-cpu/common/runtimeconfig.h"
#include "type.h"

cv::Mat drawBirdsEyeView(const size_t &points_size, const float *points_data,
                         const std::vector<vueron::BndBox> &boxes,
                         const std::vector<float> &scores,
                         const std::vector<size_t> &labels) {
  float scale = 12.0;

  int width = static_cast<int>((MAX_X_RANGE - MIN_X_RANGE) * scale);
  int height = static_cast<int>((MAX_Y_RANGE - MIN_Y_RANGE) * scale);

  cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

  // draw pcd
  for (size_t i = 0; i < points_size; ++i) {
    int x =
        static_cast<int>((points_data[i * POINT_STRIDE] - MIN_X_RANGE) * scale);
    int y = static_cast<int>((MAX_Y_RANGE - points_data[i * POINT_STRIDE + 1]) *
                             scale);
    cv::circle(image, cv::Point(x, y), 0, cv::Scalar(255, 255, 255), 1);
  }

  // draw boxes
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] < CONF_THRESH) continue;
    cv::Scalar color;
    switch (labels[i]) {
      case 1:
        color = cv::Scalar(0, 0, 255);  // red
        break;
      case 2:
        color = cv::Scalar(0, 255, 0);  // green
        break;
      case 3:
        color = cv::Scalar(255, 0, 0);  // blue
        break;
      default:
        color = cv::Scalar(0, 255, 255);  // yellow (defalut)
        break;
    }
    vueron::BndBox box = boxes[i];
    cv::Point2f center((box.x - MIN_X_RANGE) * scale,
                       (MAX_Y_RANGE - box.y) * scale);
    cv::Point2f vertices[4];
    vertices[0] = cv::Point2f(box.dx / 2, box.dy / 2);
    vertices[1] = cv::Point2f(box.dx / 2, -box.dy / 2);
    vertices[2] = cv::Point2f(-box.dx / 2, -box.dy / 2);
    vertices[3] = cv::Point2f(-box.dx / 2, box.dy / 2);

    for (int i = 0; i < 4; ++i) {
      vertices[i] = rotatePoint(vertices[i], -box.heading);
      vertices[i].x *= scale;
      vertices[i].y *= scale;
      vertices[i] += center;
    }

    for (int i = 0; i < 4; ++i) {
      cv::line(image, vertices[i], vertices[(i + 1) % 4], color, 2);
    }
  }

  return image;
}
