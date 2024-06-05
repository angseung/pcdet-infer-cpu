#include "pcdet-infer-cpu/post.h"

#include <algorithm>
#include <cassert>
#include <numeric>

#include "pcdet-infer-cpu/common/metadata.h"
#include "pcdet-infer-cpu/common/runtimeconfig.h"

void vueron::decode_to_boxes(const std::vector<std::vector<float>> &rpn_output,
                             std::vector<BndBox> &boxes,
                             std::vector<size_t> &labels,
                             std::vector<float> &scores) {
  /*
      rpn_output order: hm, dim, center, center_z, rot, iou
  */
  bool has_iou_head = (rpn_output.size() == 6);  // last had is iou head
  assert(rpn_output[0].size() ==
         CLASS_NUM * FEATURE_Y_SIZE * FEATURE_X_SIZE);                  // hm
  assert(rpn_output[1].size() == 3 * FEATURE_Y_SIZE * FEATURE_X_SIZE);  // dim
  assert(rpn_output[2].size() ==
         2 * FEATURE_Y_SIZE * FEATURE_X_SIZE);                      // center
  assert(rpn_output[3].size() == FEATURE_Y_SIZE * FEATURE_X_SIZE);  // center_z
  assert(rpn_output[4].size() == 2 * FEATURE_Y_SIZE * FEATURE_X_SIZE);  // rot
  if (has_iou_head) {
    assert(rpn_output[5].size() == FEATURE_Y_SIZE * FEATURE_X_SIZE);  // iou
  }

  size_t head_stride = GRID_Y_SIZE / FEATURE_Y_SIZE;
  std::vector<float> hm = rpn_output[0];
  assert(hm.size() == CLASS_NUM * FEATURE_Y_SIZE * FEATURE_X_SIZE);

  std::vector<size_t> indices(hm.size());
  assert(IOU_RECTIFIER.size() == 3);

  /*
      get topk scores and their indices
  */
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + NMS_PRE_MAXSIZE,
                    indices.end(),
                    [&](size_t A, size_t B) { return hm[A] > hm[B]; });

  /*
      decode into boxes
  */
  for (size_t j = 0; j < NMS_PRE_MAXSIZE; j++) {
    size_t channel_offset = FEATURE_X_SIZE * FEATURE_Y_SIZE;
    size_t idx = indices[j];  // index for hm ONLY
    size_t s_idx =
        idx % (FEATURE_X_SIZE *
               FEATURE_Y_SIZE);  // per-channel index for the other heads
    assert(idx < CLASS_NUM * FEATURE_X_SIZE * FEATURE_Y_SIZE);
    assert(s_idx < FEATURE_X_SIZE * FEATURE_Y_SIZE);

    // calc grid index
    BndBox box{};
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
    box.x = head_stride * PILLAR_X_SIZE * (grid_x + rpn_output[2][s_idx]) +
            MIN_X_RANGE;
    box.y = head_stride * PILLAR_Y_SIZE *
                (grid_y + rpn_output[2][channel_offset + s_idx]) +
            MIN_Y_RANGE;
    box.z = rpn_output[3][s_idx];

    /*
        append decoded boxes, scores, and labels
    */
    // rectifying score if model has iou head
    float rectified_score;
    if (has_iou_head) {
      rectified_score = rectify_score(sigmoid(hm[idx]), rpn_output[5][s_idx],
                                      IOU_RECTIFIER[label]);
    } else {
      rectified_score = sigmoid(hm[idx]);
    }
    if (rectified_score > SCORE_THRESH) {
      scores.push_back(rectified_score);
      boxes.push_back(box);
      labels.push_back(label + 1);
    }
  }
}

void vueron::nms(const std::vector<BndBox> &boxes,
                 const std::vector<float> &scores,
                 std::vector<bool> &suppressed, const float &iou_threshold) {
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
    if (processed >= MAX_OBJ_PER_SAMPLE) {
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

void vueron::gather_boxes(const std::vector<BndBox> &boxes,
                          const std::vector<size_t> &labels,
                          const std::vector<float> &scores,
                          std::vector<BndBox> &nms_boxes,
                          std::vector<size_t> &nms_labels,
                          std::vector<float> &nms_scores,
                          const std::vector<bool> &suppressed) {
  for (size_t j = 0; j < boxes.size(); j++) {
    if (!suppressed[j]) {
      nms_boxes.push_back(boxes[j]);
      nms_labels.push_back(labels[j]);
      nms_scores.push_back(scores[j]);

      if (nms_boxes.size() >= MAX_OBJ_PER_SAMPLE) {
        break;
      }
    }
  }
}
