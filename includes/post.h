#ifndef POST_H
#define POST_H

#include "params.h"
#include "type.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

namespace vueron {

void rectify_score(std::vector<float> hm, std::vector<float> iou) {
    assert(hm.size() / CLASS_NUM == iou.size());
    std::vector<float> rect_scores(IOU_RECTIFIER);
    float alpha = rect_scores[l];
    float iou = (*(_iou + i) + 1) * 0.5;
    *(_score + i) =
        powf(*(_score + i), (1 - alpha)) * powf(clip(iou, 0., 1.), alpha);
}

void decode_to_boxes(std::vector<std::vector<float>> &rpn_output) {
    std::vector<float> rect_scores(IOU_RECTIFIER);
    assert(rect_scores.size() == 3);
}

} // namespace vueron

#endif // POST_H
