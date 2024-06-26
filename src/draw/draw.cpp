#include "draw/draw.h"

#include <cmath>
#include <vector>

#include "pcdet-infer-cpu/common/metadata.h"
#include "pcdet-infer-cpu/common/runtimeconfig.h"
#include "type.h"

std::string floatToString(const float value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << value;
  return out.str();
}

cv::Mat drawBirdsEyeView(const size_t point_buf_len, const size_t point_stride,
                         const float *points_data,
                         const std::vector<vueron::BndBox> &boxes,
                         const std::vector<float> &scores,
                         const std::vector<size_t> &labels) {
  constexpr float scale = 12.0;
  const size_t points_size = point_buf_len / point_stride;

  const int width = static_cast<int>((MAX_X_RANGE - MIN_X_RANGE) * scale);
  const int height = static_cast<int>((MAX_Y_RANGE - MIN_Y_RANGE) * scale);

  cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

  // draw pcd
  for (size_t i = 0; i < points_size; ++i) {
    const int x =
        static_cast<int>((points_data[i * point_stride] - MIN_X_RANGE) * scale);
    const int y = static_cast<int>(
        (MAX_Y_RANGE - points_data[i * point_stride + 1]) * scale);
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
    vueron::BndBox box{boxes[i]};
    const cv::Point2f center((box.x - MIN_X_RANGE) * scale,
                             (MAX_Y_RANGE - box.y) * scale);
    cv::Point2f vertices[4];
    vertices[0] = cv::Point2f(box.dx / 2, box.dy / 2);
    vertices[1] = cv::Point2f(box.dx / 2, -box.dy / 2);
    vertices[2] = cv::Point2f(-box.dx / 2, -box.dy / 2);
    vertices[3] = cv::Point2f(-box.dx / 2, box.dy / 2);

    std::string text = floatToString(scores[i]);
    cv::putText(image, text, center, 1, 1.0f, cv::Scalar(0, 255, 255), 1);

    for (auto &vertice : vertices) {
      vertice = rotatePoint(vertice, -box.heading);
      vertice.x *= scale;
      vertice.y *= scale;
      vertice += center;
    }

    for (int j = 0; j < 4; ++j) {
      cv::line(image, vertices[j], vertices[(j + 1) % 4], color, 2);
    }
  }

  return image;
}
