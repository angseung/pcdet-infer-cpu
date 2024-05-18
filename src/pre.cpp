#include "pre.h"
#include "common.h"
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>

// 입력 데이터 크기
const size_t BATCH_SIZE = 25000;
const size_t FEATURE_SIZE1 = 20;
const size_t FEATURE_SIZE2 = 10;

namespace vueron {} // namespace vueron
