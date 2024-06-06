#ifndef __PRE_H__
#define __PRE_H__

#include <vector>

#include "pcdet-infer-cpu/common/metadata.h"
#include "pcdet-infer-cpu/common/runtimeconfig.h"

namespace vueron {

struct Pillar {
  std::vector<size_t> point_index;
  size_t pillar_grid_x;
  size_t pillar_grid_y;
  size_t point_num_in_pillar;
  bool is_empty;

  Pillar() = delete;
  Pillar(const size_t &point_num);
};

void voxelization(std::vector<Pillar> &bev_pillar, const float *points,
                  const size_t &points_buf_len, const size_t &point_stride);

size_t point_decoration(const std::vector<Pillar> &bev_pillar,
                        std::vector<size_t> &voxel_coords,
                        std::vector<size_t> &voxel_num_points,
                        std::vector<float> &pfe_input, const float *points,
                        const size_t &point_stride);

/*
  Deprecated
*/
void pfe_run(const std::vector<float> &pfe_input,
             std::vector<float> &pfe_output);

void scatter(const std::vector<float> &pfe_output,
             const std::vector<size_t> &voxel_coords, const size_t &num_pillars,
             std::vector<float> &rpn_input);

}  // namespace vueron
#endif  // __PRE_H__
