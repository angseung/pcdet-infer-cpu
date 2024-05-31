#ifndef __RPN_H__
#define __RPN_H__

#include "config.h"
#include "onnxruntime_cxx_api.h"
#include "params.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory.h>
#include <numeric>
#include <random>
#include <vector>

namespace vueron {

void rpn_run(const std::vector<float> &rpn_input,
             std::vector<std::vector<float>> &rpn_output);
} // namespace vueron

#endif // __RPN_H__
