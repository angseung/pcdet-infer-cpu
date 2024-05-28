#ifndef POST_H
#define POST_H

#include "params.h"
#include "type.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace vueron {

float clip(float val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}
float sigmoid(float x) { return 1.0f / (1.0f + log(-x)); }
float exponential(float x) { return exp(x); }

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
    size_t head_stride = GRID_Y_SIZE / FEATURE_Y_SIZE;
    std::vector<float> hm = rpn_output[0];
    assert(hm.size() == CLASS_NUM * FEATURE_Y_SIZE * FEATURE_X_SIZE);

    std::vector<size_t> indices(MAX_BOX_NUM_BEFORE_NMS);
    std::vector<float> rect_scores(IOU_RECTIFIER);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + MAX_BOX_NUM_BEFORE_NMS,
                      indices.end(),
                      [&](size_t A, size_t B) { return hm[A] > hm[B]; });

    for (size_t j = 0; j < indices.size(); j++) {
        size_t idx = indices[j];
#ifdef _DEBUG
        std::cout << indices[j] << " : " << sigmoid(hm[indices[j]])
                  << std::endl;
#endif
        BndBox box;
        size_t grid_x = idx % FEATURE_X_SIZE;
        size_t grid_y = (idx / FEATURE_X_SIZE) % FEATURE_Y_SIZE;
        size_t label = idx / (FEATURE_Y_SIZE * FEATURE_X_SIZE);

        box.dx = exponential(rpn_output[1][idx]);
        box.dy = exponential(
            rpn_output[1][FEATURE_Y_SIZE * FEATURE_X_SIZE + idx]);
        box.dz = exponential(
            rpn_output[1][2 * FEATURE_Y_SIZE * FEATURE_X_SIZE + idx]);

        float cos_rad = rpn_output[4][idx];
        float sin_rad =
            rpn_output[4][FEATURE_Y_SIZE * FEATURE_X_SIZE + idx];
        box.heading = atan2(sin_rad, cos_rad);

        box.x = head_stride * VOXEL_X_SIZE * (grid_x + rpn_output[2][idx]) +
                MIN_X_RANGE;
        box.y = head_stride * VOXEL_Y_SIZE *
                    (grid_y +
                     rpn_output[2][FEATURE_Y_SIZE * FEATURE_X_SIZE + idx]) +
                MIN_Y_RANGE;
        box.z = rpn_output[3][idx];

        float curr_iou = rpn_output[5][idx];

        boxes[j] = box;
        scores[j] =
            rectify_score(sigmoid(hm[idx]), curr_iou, rect_scores[label]);
        labels[j] = label + 1;
    }
}

} // namespace vueron

#endif // POST_H
