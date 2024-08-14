#ifndef __DEMO_COMMON_H__
#define __DEMO_COMMON_H__

#include <glob.h>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>

#include "draw/draw.h"
#include "pcdet-infer-cpu/common/metadata.h"
#include "pcdet-infer-cpu/common/runtimeconfig.h"
#include "pcl.h"

#define VEH_THRESHOLD 0.5f
#define PED_THRESHOLD 0.5f
#define CYC_THRESHOLD 0.5f

#endif  //__DEMO_COMMON_H__
