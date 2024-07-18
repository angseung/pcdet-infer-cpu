#ifndef PCDET_C_H
#define PCDET_C_H

#include <cstddef>
#include <memory>

#include "pcdet-infer-cpu/common/box.h"
#include "pcdet-infer-cpu/common/metadata.h"
#include "pcdet-infer-cpu/common/runtimeconfig.h"

const char* get_pcdet_cpu_version(void);

void pcdet_initialize(const char* metadata_path,
                      struct Runtimeconfig* runtimeconfig);

size_t pcdet_run(const float* points, int points_num, int point_stride,
                 float** score, size_t** label, Box** box);

void pcdet_finalize(void);

#endif  // PCDET_C_H
