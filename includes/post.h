#ifndef __POST_H__
#define __POST_H__

#include "params.h"
#include "type.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace vueron {

const float EPS = 1e-8;

struct Point {
    float x, y;
    Point() {}
    Point(double _x, double _y) { x = _x, y = _y; }

    void set(float _x, float _y) {
        x = _x;
        y = _y;
    }

    Point operator+(const Point &b) const { return Point(x + b.x, y + b.y); }

    Point operator-(const Point &b) const { return Point(x - b.x, y - b.y); }
};

inline float min(float a, float b) { return a > b ? b : a; }

inline float max(float a, float b) { return a > b ? a : b; }

inline float cross(const Point &a, const Point &b) {
    return a.x * b.y - a.y * b.x;
}

inline float cross(const Point &p1, const Point &p2, const Point &p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int check_rect_cross(const Point &p1, const Point &p2, const Point &q1,
                            const Point &q2) {
    int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
              min(q1.x, q2.x) <= max(p1.x, p2.x) &&
              min(p1.y, p2.y) <= max(q1.y, q2.y) &&
              min(q1.y, q2.y) <= max(p1.y, p2.y);
    return ret;
}

inline int check_in_box2d(const float *box, const Point &p) {
    // params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;

    float center_x = box[0], center_y = box[1];
    float angle_cos = cos(-box[6]),
          angle_sin =
              sin(-box[6]); // rotate the point in the opposite direction of box
    float rot_x =
        (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3] / 2 + MARGIN &&
            fabs(rot_y) < box[4] / 2 + MARGIN);
}

inline void rotate_around_center(const Point &center, const float angle_cos,
                                 const float angle_sin, Point &p) {
    float new_x = (p.x - center.x) * angle_cos +
                  (p.y - center.y) * (-angle_sin) + center.x;
    float new_y =
        (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

inline int point_cmp(const Point &a, const Point &b, const Point &center) {
    return atan2(a.y - center.y, a.x - center.x) >
           atan2(b.y - center.y, b.x - center.x);
}

inline int intersection(const Point &p1, const Point &p0, const Point &q1,
                        const Point &q0, Point &ans) {
    // fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0)
        return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0))
        return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > EPS) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    } else {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x,
              c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x,
              c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

inline float clip(float val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

inline float rectify_score(float score, float iou, float alpha) {
    float new_iou = (iou + 1.0f) * 0.5f;
    float new_score =
        powf(score, (1.0f - alpha)) * powf(clip(new_iou, 0.0f, 1.0f), alpha);
    return new_score;
}

void decode_to_boxes(const std::vector<std::vector<float>> &rpn_output,
                     std::vector<BndBox> &boxes, std::vector<size_t> &labels,
                     std::vector<float> &scores) {
    /*
        rpn_output order: hm, dim, center, center_z, rot, iou
    */
    assert(rpn_output[0].size() ==
           CLASS_NUM * FEATURE_Y_SIZE * FEATURE_X_SIZE);                 // hm
    assert(rpn_output[1].size() == 3 * FEATURE_Y_SIZE * FEATURE_X_SIZE); // dim
    assert(rpn_output[2].size() ==
           2 * FEATURE_Y_SIZE * FEATURE_X_SIZE);                     // center
    assert(rpn_output[3].size() == FEATURE_Y_SIZE * FEATURE_X_SIZE); // center_z
    assert(rpn_output[4].size() == 2 * FEATURE_Y_SIZE * FEATURE_X_SIZE); // rot
    assert(rpn_output[5].size() == FEATURE_Y_SIZE * FEATURE_X_SIZE);     // iou

    size_t head_stride = GRID_Y_SIZE / FEATURE_Y_SIZE;
    std::vector<float> hm = rpn_output[0];
    assert(hm.size() == CLASS_NUM * FEATURE_Y_SIZE * FEATURE_X_SIZE);

    std::vector<size_t> indices(hm.size());
    std::vector<float> rect_scores(IOU_RECTIFIER);
    assert(rect_scores.size() == 3);

    /*
        get topk scores and their indices
    */
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + MAX_BOX_NUM_BEFORE_NMS,
                      indices.end(),
                      [&](size_t A, size_t B) { return hm[A] > hm[B]; });

    /*
        decode into boxes
    */
    for (size_t j = 0; j < MAX_BOX_NUM_BEFORE_NMS; j++) {
        size_t channel_offset = FEATURE_X_SIZE * FEATURE_Y_SIZE;
        size_t idx = indices[j]; // index for hm ONLY
        size_t s_idx =
            idx % (FEATURE_X_SIZE *
                   FEATURE_Y_SIZE); // per-channel index for the other heads
        assert(idx < CLASS_NUM * FEATURE_X_SIZE * FEATURE_Y_SIZE);
        assert(s_idx < FEATURE_X_SIZE * FEATURE_Y_SIZE);

        // calc grid index
        BndBox box;
        size_t label = idx / channel_offset;
        size_t grid_x = idx % FEATURE_X_SIZE;
        size_t grid_y = (idx / FEATURE_X_SIZE) % FEATURE_Y_SIZE;
        assert(grid_x < FEATURE_X_SIZE);
        assert(grid_y < FEATURE_Y_SIZE);
        assert(label >= 0 && label < FEATURE_NUM);

        // calc box dimensions
        box.dx = exp(rpn_output[1][s_idx]);
        box.dy = exp(rpn_output[1][channel_offset + s_idx]);
        box.dz = exp(rpn_output[1][2 * channel_offset + s_idx]);

        // calc heading angle in radian
        float cos_rad = rpn_output[4][s_idx];
        float sin_rad = rpn_output[4][channel_offset + s_idx];
        box.heading = atan2(sin_rad, cos_rad);
        assert(box.heading <= 180.0 / M_PI && box.heading >= -180.0 / M_PI);

        // calc center point
        box.x = head_stride * VOXEL_X_SIZE * (grid_x + rpn_output[2][s_idx]) +
                MIN_X_RANGE;
        box.y = head_stride * VOXEL_Y_SIZE *
                    (grid_y + rpn_output[2][channel_offset + s_idx]) +
                MIN_Y_RANGE;
        box.z = rpn_output[3][s_idx];

        /*
            append decoded boxes, scores, and labels
        */
        // rectifying score if model has iou head
        float rectified_score = rectify_score(
            sigmoid(hm[idx]), rpn_output[5][s_idx], rect_scores[label]);
        if (rectified_score > SCORE_THRESH) {
            scores.push_back(rectified_score);
            boxes.push_back(box);
            labels.push_back(label + 1);
        }
    }
}
inline float box_overlap(const float *box_a, const float *box_b) {
    // params: box_a (7) [x, y, z, dx, dy, dz, heading]
    // params: box_b (7) [x, y, z, dx, dy, dz, heading]

    //    float a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[2], a_y2 =
    //    box_a[3], a_angle = box_a[4]; float b_x1 = box_b[0], b_y1 = box_b[1],
    //    b_x2 = box_b[2], b_y2 = box_b[3], b_angle = box_b[4];
    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2,
          a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    Point center_a(box_a[0], box_a[1]);
    Point center_b(box_b[0], box_b[1]);

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin,
                             box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin,
                             box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
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
    Point temp;
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
    float dist = sqrt((box_a[0] - box_b[0]) * (box_a[0] - box_b[0]) +
                      (box_a[1] - box_b[1]) * (box_a[1] - box_b[1]));
    if (dist > 10.0f) {
        return 0.0f;
    }
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

void nms(const std::vector<BndBox> &boxes, const std::vector<float> &scores,
         std::vector<bool> &suppressed, float iou_threshold) {
    assert(boxes.size() == scores.size());
    // sort boxes based on their scores (descending order)
    std::vector<size_t> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return scores[a] > scores[b]; });

    size_t processed = 0;

    // Loop over each box index
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (suppressed[idx]) {
            continue;
        }

        processed++;
        if (processed >= MAX_BOX_NUM_AFTER_NMS) {
            break;
        }

        // Compare this box to the rest of the boxes
        for (size_t j = i + 1; j < indices.size(); ++j) {
            size_t idx_j = indices[j];
            if (suppressed[idx_j]) {
                continue;
            }

            // Calculate the IOU of the current box with the rest of the boxes
            if (calculateIOU((float *)&boxes[idx], (float *)&boxes[idx_j]) >
                iou_threshold) {
                suppressed[idx_j] = true;
            }
        }
    }
}

void gather_boxes(const std::vector<BndBox> &boxes,
                  const std::vector<float> &scores,
                  const std::vector<size_t> &labels,
                  std::vector<BndBox> &nms_boxes,
                  std::vector<float> &nms_scores,
                  std::vector<size_t> &nms_labels,
                  const std::vector<bool> &suppressed) {

    for (size_t j = 0; j < boxes.size(); j++) {
        if (!suppressed[j]) {
            nms_boxes.push_back(boxes[j]);
            nms_labels.push_back(labels[j]);
            nms_scores.push_back(scores[j]);

            if (nms_boxes.size() >= MAX_BOX_NUM_AFTER_NMS) {
                break;
            }
        }
    }
}

} // namespace vueron

#endif // __POST_H__
