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
        size_t idx = indices[j];
        size_t s_idx = idx % (FEATURE_X_SIZE * FEATURE_Y_SIZE);
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
        float curr_iou = rpn_output[5][s_idx];
        // rectifying score if model has iou head
        scores[j] =
            rectify_score(sigmoid(hm[idx]), curr_iou, rect_scores[label]);
        boxes[j] = box;
        labels[j] = label + 1;
    }
}

} // namespace vueron

#endif // POST_H
