#include "draw/draw.h"

#include <demo_common.h>

#include <cmath>
#include <vector>

#include "pcdet-infer-cpu/common/box.h"
#include "pcdet-infer-cpu/common/metadata.h"
#include "pcdet-infer-cpu/common/runtimeconfig.h"

#ifdef ENABLE_OPEN3D
#include <open3d/Open3D.h>

#include <Eigen/Dense>
#endif  // ENABLE_OPEN3D

std::string floatToString(const float value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << value;
  return out.str();
}

void drawBirdsEyeView(const size_t point_buf_len, const size_t point_stride,
                      const float *points_data, const std::vector<Box> &boxes,
                      const std::vector<float> &scores,
                      const std::vector<size_t> &labels, const float scale,
                      cv::Mat &image) {
  const size_t points_size = point_buf_len / point_stride;

  // draw pcd
  for (size_t i = 0; i < points_size; ++i) {
    const int x =
        static_cast<int>((points_data[i * point_stride] - MIN_X_RANGE) * scale);
    const int y = static_cast<int>(
        (MAX_Y_RANGE - points_data[i * point_stride + 1]) * scale);
    cv::circle(image, cv::Point{x, y}, 0, cv::Scalar{255, 255, 255}, 1);
  }

  // draw boxes
  const std::vector<float> score_threshold{VEH_THRESHOLD, PED_THRESHOLD,
                                           CYC_THRESHOLD};
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] < score_threshold[labels[i]]) continue;
    cv::Scalar color;
    switch (labels[i]) {
      case 0:
        color = cv::Scalar{0, 0, 255};  // red
        break;
      case 1:
        color = cv::Scalar{0, 255, 0};  // green
        break;
      case 2:
        color = cv::Scalar{255, 0, 0};  // blue
        break;
      default:
        color = cv::Scalar{0, 255, 255};  // yellow (defalut)
        break;
    }
    Box box{boxes[i]};
    const cv::Point2f center((box.x - MIN_X_RANGE) * scale,
                             (MAX_Y_RANGE - box.y) * scale);
    cv::Point2f vertices[4];
    vertices[0] = cv::Point2f(box.dx / 2, box.dy / 2);
    vertices[1] = cv::Point2f(box.dx / 2, -box.dy / 2);
    vertices[2] = cv::Point2f(-box.dx / 2, -box.dy / 2);
    vertices[3] = cv::Point2f(-box.dx / 2, box.dy / 2);

    std::string text = floatToString(scores[i]);
    cv::putText(image, text, center, 1, 1.0f, cv::Scalar{0, 255, 255}, 1);

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
}

#ifdef ENABLE_OPEN3D
void draw3DView(const size_t point_buf_len, const size_t point_stride,
                const float *points_data, const std::vector<Box> &boxes,
                const std::vector<float> &scores,
                const std::vector<size_t> &labels) {
  const size_t points_size = point_buf_len / point_stride;
  open3d::geometry::PointCloud pointcloud;
  for (size_t i = 0; i < points_size; i++) {
    pointcloud.points_.emplace_back(points_data[point_stride * i],
                                    points_data[point_stride * i + 1],
                                    points_data[point_stride * i + 2]);
  }

  const std::vector<Eigen::Vector3d> colors(points_size,
                                            Eigen::Vector3d(1.0, 1.0, 1.0));
  pointcloud.colors_ = colors;
  open3d::visualization::Visualizer visualizer;
  const std::shared_ptr<open3d::geometry::PointCloud> pointcloud_ptr(
      new open3d::geometry::PointCloud);
  *pointcloud_ptr = pointcloud;
  visualizer.CreateVisualizerWindow("Open3D", 1600, 900);
  visualizer.AddGeometry(pointcloud_ptr);

  for (size_t i = 0; i < scores.size(); i++) {
    // Define the center of the cuboid
    const Box box{boxes[i]};
    Eigen::Vector3d center(box.x, box.y, box.z);

    // Define the 3D dimensions of the cuboid
    Eigen::Vector3d dimensions(box.dx, box.dy, box.dz);

    // Define the rotation angles in radians
    const float theta = box.heading / 4;
    Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    // Generate Oriented Bounding Box
    open3d::geometry::OrientedBoundingBox obb;
    obb.center_ = center;
    obb.extent_ = dimensions;
    obb.R_ = rotation;
    auto line_set =
        open3d::geometry::LineSet::CreateFromOrientedBoundingBox(obb);
    Eigen::Vector3d color(1.0, 0.0, 0.0);  // RGB: Red
    line_set->colors_.resize(line_set->lines_.size(), color);

    visualizer.AddGeometry(line_set);
  }

  visualizer.GetRenderOption().background_color_ =
      Eigen::Vector3d(0.0, 0.0, 0.0);
  visualizer.Run();
  visualizer.DestroyVisualizerWindow();
}
#endif  // ENABLE_OPEN3D
