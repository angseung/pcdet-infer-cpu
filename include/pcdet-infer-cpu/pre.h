#ifndef __PRE_H__
#define __PRE_H__

#include <vector>

#include "params.h"
#include "type.h"

namespace vueron {

void voxelization(std::vector<Pillar> &bev_pillar, const float *points,
                  const size_t &points_buf_len, const size_t &point_stride);

size_t point_decoration(const std::vector<Pillar> &bev_pillar,
                        std::vector<size_t> &voxel_coords,
                        std::vector<size_t> &voxel_num_points,
                        std::vector<float> &pfe_input, const float *points,
                        const size_t &point_stride);

void pfe_run(const std::vector<float> &pfe_input,
             std::vector<float> &pfe_output);

void scatter(const std::vector<float> &pfe_output,
             const std::vector<size_t> &voxel_coords, const size_t &num_pillars,
             std::vector<float> &rpn_input);

}  // namespace vueron
#endif  // __PRE_H__
