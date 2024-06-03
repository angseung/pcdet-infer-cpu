#include "pcdet-infer-cpu/model.h"
#include "params.h"
#include "pcdet-infer-cpu/post.h"
#include "pcdet-infer-cpu/pre.h"
#include "pcdet-infer-cpu/rpn.h"
#include <cstdlib>

void vueron::run_model(const float *points, size_t point_buf_len,
                       size_t point_stride, std::vector<BndBox> &boxes,
                       std::vector<size_t> &labels,
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
    std::vector<BndBox> pre_boxes;  // boxes before NMS
    std::vector<size_t> pre_labels; // labels before NMS
    std::vector<float> pre_scores;  // scores before NMS

    /*
        Buffers for Final Predictions
    */
    pre_boxes.reserve(MAX_BOX_NUM_BEFORE_NMS);
    pre_labels.reserve(MAX_BOX_NUM_BEFORE_NMS);
    pre_scores.reserve(MAX_BOX_NUM_BEFORE_NMS);

    /*
        Preprocessing
    */
    voxelization(bev_pillar, points, point_buf_len, point_stride);
    size_t num_pillars =
        point_decoration(bev_pillar, voxel_coords, voxel_num_points, pfe_input,
                         points, point_stride);

    /*
        Pillar Feature Extraction
    */
    pfe_run(pfe_input, pfe_output);

    /*
        Pillar Feature Scatter
    */
    scatter(pfe_output, voxel_coords, num_pillars, bev_image);

    /*
        Backbone & Center Head
    */
    rpn_run(bev_image, rpn_outputs);

    /*
        Postprocessing
    */
    decode_to_boxes(rpn_outputs, pre_boxes, pre_labels, pre_scores);
    std::vector<bool> suppressed(pre_boxes.size(), false); // mask for nms
    nms(pre_boxes, pre_scores, suppressed, IOU_THRESH);
    gather_boxes(pre_boxes, pre_labels, pre_scores, boxes, labels, scores,
                 suppressed);
}
