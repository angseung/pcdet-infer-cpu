#ifndef __POST_H__
#define __POST_H__

#include <cmath>
#include <vector>

#include "pcdet-infer-cpu/common/metadata.h"
#include "pcdet-infer-cpu/common/runtimeconfig.h"
#include "type.h"

namespace vueron {
template <typename T>
struct Point {
  T x, y;
  Point() = default;
  Point(const T _x, const T _y) { x = _x, y = _y; }
  ~Point() = default;

  void set(const T _x, const T _y) {
    x = _x;
    y = _y;
  }

  Point operator+(const Point &b) const { return {x + b.x, y + b.y}; }

  Point operator-(const Point &b) const { return {x - b.x, y - b.y}; }
};

constexpr float EPS = 1e-8;

inline float min(const float a, const float b) { return a > b ? b : a; }

inline float max(const float a, const float b) { return a > b ? a : b; }

inline float cross(const Point<float> &a, const Point<float> &b) {
  return a.x * b.y - a.y * b.x;
}

inline float cross(const Point<float> &p1, const Point<float> &p2,
                   const Point<float> &p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int check_rect_cross(const Point<float> &p1, const Point<float> &p2,
                            const Point<float> &q1, const Point<float> &q2) {
  const int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
                  min(q1.x, q2.x) <= max(p1.x, p2.x) &&
                  min(p1.y, p2.y) <= max(q1.y, q2.y) &&
                  min(q1.y, q2.y) <= max(p1.y, p2.y);
  return ret;
}

inline int check_in_box2d(const float *box, const Point<float> &p) {
  // params: (7) [x, y, z, dx, dy, dz, heading]
  constexpr float MARGIN = 1e-2;

  const float center_x = box[0], center_y = box[1];
  const float angle_cos = cos(-box[6]);
  const float angle_sin =
      sin(-box[6]);  // rotate the point in the opposite direction of box
  const float rot_x =
      (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
  const float rot_y =
      (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

  return (fabs(rot_x) < box[3] / 2 + MARGIN &&
          fabs(rot_y) < box[4] / 2 + MARGIN);
}

inline void rotate_around_center(const Point<float> &center,
                                 const float angle_cos, const float angle_sin,
                                 Point<float> &p) {
  const float new_x =
      (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
  const float new_y =
      (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
  p.set(new_x, new_y);
}

inline int point_cmp(const Point<float> &a, const Point<float> &b,
                     const Point<float> &center) {
  return atan2(a.y - center.y, a.x - center.x) >
         atan2(b.y - center.y, b.x - center.x);
}

inline int intersection(const Point<float> &p1, const Point<float> &p0,
                        const Point<float> &q1, const Point<float> &q0,
                        Point<float> &ans) {
  // fast exclusion
  if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

  // check cross standing
  const float s1 = cross(q0, p1, p0);
  const float s2 = cross(p1, q1, p0);
  const float s3 = cross(p0, q1, q0);
  const float s4 = cross(q1, p1, q0);

  if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

  // calculate intersection of two lines
  const float s5 = cross(q1, p1, p0);
  if (fabs(s5 - s1) > EPS) {
    ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
    ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

  } else {
    const float a0 = p0.y - p1.y, b0 = p1.x - p0.x,
                c0 = p0.x * p1.y - p1.x * p0.y;
    const float a1 = q0.y - q1.y, b1 = q1.x - q0.x,
                c1 = q0.x * q1.y - q1.x * q0.y;
    const float D = a0 * b1 - a1 * b0;

    ans.x = (b0 * c1 - b1 * c0) / D;
    ans.y = (a1 * c0 - a0 * c1) / D;
  }

  return 1;
}

inline float clip(const float val, const float min_val, const float max_val) {
  return fminf(fmaxf(val, min_val), max_val);
}
inline float sigmoid(const float x) { return 1.0f / (1.0f + exp(-x)); }

inline float rectify_score(const float score, const float iou,
                           const float alpha) {
  const float new_iou = (iou + 1.0f) * 0.5f;
  const float new_score =
      powf(score, (1.0f - alpha)) * powf(clip(new_iou, 0.0f, 1.0f), alpha);
  return new_score;
}

void decode_to_boxes(const std::vector<std::vector<float>> &rpn_output,
                     std::vector<BndBox> &boxes, std::vector<size_t> &labels,
                     std::vector<float> &scores);

inline float box_overlap(const float *box_a, const float *box_b) {
  // params: box_a (7) [x, y, z, dx, dy, dz, heading]
  // params: box_b (7) [x, y, z, dx, dy, dz, heading]

  const float a_angle = box_a[6], b_angle = box_b[6];
  const float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2,
              a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
  const float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
  const float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
  const float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
  const float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

  const Point<float> center_a(box_a[0], box_a[1]);
  const Point<float> center_b(box_b[0], box_b[1]);

  Point<float> box_a_corners[5];
  box_a_corners[0].set(a_x1, a_y1);
  box_a_corners[1].set(a_x2, a_y1);
  box_a_corners[2].set(a_x2, a_y2);
  box_a_corners[3].set(a_x1, a_y2);

  Point<float> box_b_corners[5];
  box_b_corners[0].set(b_x1, b_y1);
  box_b_corners[1].set(b_x2, b_y1);
  box_b_corners[2].set(b_x2, b_y2);
  box_b_corners[3].set(b_x1, b_y2);

  // get oriented corners
  const float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
  const float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

  for (int k = 0; k < 4; k++) {
    rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
    rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
  }

  box_a_corners[4] = box_a_corners[0];
  box_b_corners[4] = box_b_corners[0];

  // get intersection of lines
  Point<float> cross_points[16];
  Point<float> poly_center;
  int cnt = 0, flag = 0;

  poly_center.set(0, 0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                          box_b_corners[j + 1], box_b_corners[j],
                          cross_points[cnt]);
      if (flag) {
        poly_center = poly_center + cross_points[cnt];
        cnt++;
      }
    }
  }

  // check corners
  for (int k = 0; k < 4; k++) {
    if (check_in_box2d(box_a, box_b_corners[k])) {
      poly_center = poly_center + box_b_corners[k];
      cross_points[cnt] = box_b_corners[k];
      cnt++;
    }
    if (check_in_box2d(box_b, box_a_corners[k])) {
      poly_center = poly_center + box_a_corners[k];
      cross_points[cnt] = box_a_corners[k];
      cnt++;
    }
  }

  poly_center.x /= cnt;
  poly_center.y /= cnt;

  // sort the points of polygon
  Point<float> temp{};
  for (int j = 0; j < cnt - 1; j++) {
    for (int i = 0; i < cnt - j - 1; i++) {
      if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
        temp = cross_points[i];
        cross_points[i] = cross_points[i + 1];
        cross_points[i + 1] = temp;
      }
    }
  }

  // get the overlap areas
  float area = 0;
  for (int k = 0; k < cnt - 1; k++) {
    area += cross(cross_points[k] - cross_points[0],
                  cross_points[k + 1] - cross_points[0]);
  }

  return fabs(area) / 2.0;
}

inline float calculateIOU(const float *box_a, const float *box_b) {
  // params: box_a (7) [x, y, z, dx, dy, dz, heading]
  // params: box_b (7) [x, y, z, dx, dy, dz, heading]

  /*
      skip if distance of two boxes are large enough
  */
  const float dist = sqrt((box_a[0] - box_b[0]) * (box_a[0] - box_b[0]) +
                          (box_a[1] - box_b[1]) * (box_a[1] - box_b[1]));
  if (dist > PRE_NMS_DISTANCE_THD) {
    return 0.0f;
  }
  const float sa = box_a[3] * box_a[4];
  const float sb = box_b[3] * box_b[4];
  const float s_overlap = box_overlap(box_a, box_b);
  return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

void nms(const std::vector<BndBox> &boxes, const std::vector<float> &scores,
         std::vector<bool> &suppressed, const float &iou_threshold);

void gather_boxes(const std::vector<BndBox> &boxes,
                  const std::vector<size_t> &labels,
                  const std::vector<float> &scores,
                  std::vector<BndBox> &nms_boxes,
                  std::vector<size_t> &nms_labels,
                  std::vector<float> &nms_scores,
                  const std::vector<bool> &suppressed);

}  // namespace vueron

#endif  // __POST_H__
