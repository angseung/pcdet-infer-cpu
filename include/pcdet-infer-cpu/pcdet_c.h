#ifndef __PCDET_C_H__
#define __PCDET_C_H__

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

#include "common/metadata.h"
#include "common/runtimeconfig.h"
#include "common/type.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* get_pcdet_cpu_version(void);

void pcdet_initialize(const char* metadata_path,
                      const struct RuntimeConfig* runtimeconfig);

size_t pcdet_run(const float* points, int point_buf_len, int point_stride,
                 PredBox** box);

void pcdet_finalize(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __PCDET_C_H__
