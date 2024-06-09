#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>

#include "type.h"

namespace vueron {
void run_model(const float *points, const size_t &point_buf_len,
               const size_t &point_stride, std::vector<BndBox> &boxes,
               std::vector<size_t> &labels, std::vector<float> &scores);

}  // namespace vueron

#endif  // __MODEL_H__
