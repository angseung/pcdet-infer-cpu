#ifndef __MODEL_H__
#define __MODEL_H__

#include "type.h"
#include <cstdlib>
#include <vector>

namespace vueron {
void run_model(const float *points, size_t point_buf_len, size_t point_stride,
               std::vector<BndBox> &boxes, std::vector<float> &scores,
               std::vector<size_t> &labels);

} // namespace vueron

#endif // __MODEL_H__