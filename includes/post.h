#ifndef POST_H
#define POST_H

#include "params.h"
#include <cassert>
#include <iostream>
#include <vector>

namespace vueron {

void rectify_score(std::vector<float> hm, std::vector<float> iou) {
    assert(hm.size() / CLASS_NUM == iou.size());
}
} // namespace vueron

#endif // POST_H
