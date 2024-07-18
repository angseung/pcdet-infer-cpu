#ifndef PCDET_C_H
#define PCDET_C_H

#include "pcdet-infer-cpu/common/box.h"
#include "pcdet-infer-cpu/common/metadata.h"
#include "pcdet-infer-cpu/common/runtimeconfig.h"
#include <memory>
#include <cstddef>

const char* get_pcdet_cpu_version(void);

void pcdet_initialize(const char* metadata_path, struct Runtimeconfig* runtimeconfig);

int pcdet_run();

void pcdet_finalize();

#endif //PCDET_C_H
