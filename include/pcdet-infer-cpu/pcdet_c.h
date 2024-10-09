#ifndef __PCDET_C_H__
#define __PCDET_C_H__

#ifdef __cplusplus
#include <cstddef>
#include <string>
#include <vector>
#else
#include <stddef.h>
#endif

#include "common/box.h"
#include "common/metadata.h"
#include "common/runtimeconfig.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* GetlibDLVersion(void);

const char* GetCUDATRTVersion(void);

// Use "struct" keyword for compatibility with C.

void pcdet_initialize(const char* onnx_file, const char* onnx_hash,
                      const struct Runtimeconfig runtimeconfig);

void pcdet_initialize_with_metadata(const char* metadata_path,
                                    const char* onnx_hash,
                                    const struct RuntimeConfig* runtimeconfig);

int pcdet_infer(size_t points_size, const float* points, struct Bndbox** boxes);

void pcdet_finalize(void);

#ifdef __cplusplus
}  // extern "C"
#endif

// Interface for "C++"
#ifdef __cplusplus

std::vector<Bndbox> pcdet_infer(size_t points_size, const float* points);

#endif

#endif  // __PCDET_C_H__
