#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include "params.h"
#include "post.h"
#include "pre.h"
#include "rpn.h"

namespace vueron {

void run_model(const float *points, size_t point_buf_len, size_t point_stride,
               std::vector<BndBox> &boxes, std::vector<size_t> &labels,
               std::vector<float> &scores) {
    std::vector<Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
    std::vector<size_t> voxel_coords; // (x, y)
    std::vector<size_t> voxel_num_points;
    std::vector<float> pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR *
                                     FEATURE_NUM,
                                 0.0f); // input of pfe_run()
    std::vector<float> pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER,
                                  0.0f); // input of scatter()
    std::vector<float> bev_image(GRID_Y_SIZE * GRID_X_SIZE *
                                     NUM_FEATURE_SCATTER,
                                 0.0f); // input of RPN
    std::vector<std::vector<float>> rpn_outputs;

    voxelization(bev_pillar, points, point_buf_len, point_stride);
    size_t num_pillars =
        point_decoration(bev_pillar, voxel_coords, voxel_num_points, pfe_input,
                         points, point_buf_len, point_stride);
    pfe_run(pfe_input, pfe_output);
    scatter(pfe_output, voxel_coords, num_pillars, bev_image);
    rpn_run(bev_image, rpn_outputs);
    decode_to_boxes(rpn_outputs, boxes, labels, scores);
}

} // namespace vueron
#endif