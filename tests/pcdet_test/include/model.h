#ifndef __MODEL_H_TEST__
#define __MODEL_H_TEST__

#include <cstddef>
#include <vector>

#include "pcdet-infer-cpu/common/type.h"

namespace vueron {
void run_model(const float *points, size_t point_buf_len, size_t point_stride,
               std::vector<BndBox> &boxes, std::vector<size_t> &labels,
               std::vector<float> &scores);

}  // namespace vueron

#endif  // __MODEL_H_TEST__
