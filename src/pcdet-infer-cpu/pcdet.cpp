#include "pcdet-infer-cpu/pcdet.h"
#include "pcdet-infer-cpu/post.h"
#include "pcdet-infer-cpu/pre.h"
#include <chrono>
#include <config.h>
#include <iomanip>
#include <iostream>

vueron::PCDet::PCDet()
    : bev_pillar(GRID_Y_SIZE * GRID_X_SIZE),
      pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f),
      pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER, 0.0f),
      bev_image(GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER, 0.0f),
      suppressed(MAX_BOX_NUM_BEFORE_NMS, false), num_pillars(0),
      pfe_path(PFE_PATH),
      pfe_input_dim({MAX_VOXELS, MAX_NUM_POINTS_PER_PILLAR, FEATURE_NUM}),
      pfe(pfe_path, pfe_input_dim,
          MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM),
      rpn_path(RPN_PATH),
      rpn_input_dim({1, NUM_FEATURE_SCATTER, GRID_Y_SIZE, GRID_X_SIZE}),
      rpn(rpn_path, rpn_input_dim,
          GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER) {
    std::cout << "PFE Model Initialized with default path, " << PFE_PATH
              << std::endl;
    std::cout << "RPN Model Initialized with default path, " << RPN_PATH
              << std::endl;
};

vueron::PCDet::PCDet(std::string pfe_path, std::string rpn_path)
    : bev_pillar(GRID_Y_SIZE * GRID_X_SIZE),
      pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f),
      pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER, 0.0f),
      bev_image(GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER, 0.0f),
      suppressed(MAX_BOX_NUM_BEFORE_NMS, false), num_pillars(0),
      pfe_path(pfe_path),
      pfe_input_dim{MAX_VOXELS, MAX_NUM_POINTS_PER_PILLAR, FEATURE_NUM},
      pfe(pfe_path, pfe_input_dim,
          MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM),
      rpn_path(rpn_path),
      rpn_input_dim{1, NUM_FEATURE_SCATTER, GRID_Y_SIZE, GRID_X_SIZE},
      rpn(rpn_path, rpn_input_dim,
          GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER) {
    std::cout << "PFE Model Initialized with " << PFE_PATH << std::endl;
    std::cout << "RPN Model Initialized with " << RPN_PATH << std::endl;
};

vueron::PCDet::~PCDet(){};

void vueron::PCDet::preprocess(const float *points, const size_t point_buf_len,
                               const size_t point_stride) {
    vueron::voxelization(bev_pillar, points, point_buf_len, point_stride);
    num_pillars =
        vueron::point_decoration(bev_pillar, voxel_coords, voxel_num_points,
                                 pfe_input, points, point_stride);
}

void vueron::PCDet::scatter(void) {
    vueron::scatter(pfe_output, voxel_coords, num_pillars, bev_image);
}

void vueron::PCDet::postprocess(std::vector<vueron::BndBox> &final_boxes,
                                std::vector<size_t> &final_labels,
                                std::vector<float> &final_scores) {
    vueron::decode_to_boxes(rpn_outputs, pre_boxes, pre_labels, pre_scores);
    vueron::nms(pre_boxes, pre_scores, suppressed, IOU_THRESH);
    vueron::gather_boxes(pre_boxes, pre_labels, pre_scores, final_boxes,
                         final_labels, final_scores, suppressed);
}

void vueron::PCDet::get_pred(std::vector<PredBox> &boxes) {
    for (size_t i = 0; i < post_boxes.size(); i++) {
        PredBox box{0.0f};
        box.x = post_boxes[i].x;
        box.y = post_boxes[i].y;
        box.z = post_boxes[i].z;
        box.dx = post_boxes[i].dx;
        box.dy = post_boxes[i].dy;
        box.dz = post_boxes[i].dz;
        box.heading = post_boxes[i].heading;
        box.score = post_scores[i];
        box.label = post_labels[i];

        boxes.push_back(box);
    }
}

void vueron::PCDet::do_infer(const float *points, const size_t point_buf_len,
                             const size_t point_stride,
                             std::vector<PredBox> &final_boxes) {
    /**
     * @brief
     * It writes predictions into a vector, boxes.
     *
     */
    auto pre_startTime = std::chrono::system_clock::now();
    vueron::PCDet::preprocess(points, point_buf_len, point_stride);
    auto pre_endTime = std::chrono::system_clock::now();
    auto pre_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        pre_endTime - pre_startTime);

    auto pfe_startTime = std::chrono::system_clock::now();
    pfe.run(pfe_input, pfe_output);
    auto pfe_endTime = std::chrono::system_clock::now();
    auto pfe_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        pfe_endTime - pfe_startTime);

    auto scatter_startTime = std::chrono::system_clock::now();
    vueron::PCDet::scatter();
    auto scatter_endTime = std::chrono::system_clock::now();
    auto scatter_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        scatter_endTime - scatter_startTime);

    auto rpn_startTime = std::chrono::system_clock::now();
    rpn.run(bev_image, rpn_outputs);
    auto rpn_endTime = std::chrono::system_clock::now();
    auto rpn_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        rpn_endTime - rpn_startTime);

    auto post_startTime = std::chrono::system_clock::now();
    vueron::PCDet::postprocess(post_boxes, post_labels, post_scores);
    auto post_endTime = std::chrono::system_clock::now();
    auto post_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        post_endTime - post_startTime);

    auto gather_boxes_startTime = std::chrono::system_clock::now();
    vueron::PCDet::get_pred(final_boxes);
    auto gather_boxes_endTime = std::chrono::system_clock::now();
    auto gather_boxes_sec =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            gather_boxes_endTime - gather_boxes_startTime);

#ifdef _PROFILE
    std::cout << "Preprocessing: " << pre_sec.count() / 1000.0 << std::endl;
    std::cout << "PFE: " << pfe_sec.count() / 1000.0 << std::endl;
    std::cout << "Scatter: " << scatter_sec.count() / 1000.0 << std::endl;
    std::cout << "RPN: " << rpn_sec.count() / 1000.0 << std::endl;
    std::cout << "Postprocessing: " << post_sec.count() / 1000.0 << std::endl;
    std::cout << "Gather Boxes: " << gather_boxes_sec.count() / 1000.0
              << std::endl;
#endif

    /*
        Reset buffers
    */
    std::fill(bev_pillar.begin(), bev_pillar.end(), Pillar());
    std::fill(pfe_input.begin(), pfe_input.end(), 0.0f);
    std::fill(pfe_output.begin(), pfe_output.end(), 0.0f);
    std::fill(bev_image.begin(), bev_image.end(), 0.0f);
    std::fill(suppressed.begin(), suppressed.end(), false);
    num_pillars = 0;

    /*
        Clear buffers to have zero length
    */
    voxel_coords.clear();
    voxel_num_points.clear();
    rpn_outputs.clear();
    pre_boxes.clear();
    pre_labels.clear();
    pre_scores.clear();
    post_boxes.clear();
    post_labels.clear();
    post_scores.clear();
};

void vueron::PCDet::do_infer(const float *points, const size_t point_buf_len,
                             const size_t point_stride,
                             std::vector<vueron::BndBox> &final_boxes,
                             std::vector<size_t> &final_labels,
                             std::vector<float> &final_scores) {
    /**
     * @brief
     * It writes predictions into three vectors, final_boxes, final_labels, and
     * final_scores.
     *
     */
    auto pre_startTime = std::chrono::system_clock::now();
    vueron::PCDet::preprocess(points, point_buf_len, point_stride);
    auto pre_endTime = std::chrono::system_clock::now();
    auto pre_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        pre_endTime - pre_startTime);

    auto pfe_startTime = std::chrono::system_clock::now();
    pfe.run(pfe_input, pfe_output);
    auto pfe_endTime = std::chrono::system_clock::now();
    auto pfe_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        pfe_endTime - pfe_startTime);

    auto scatter_startTime = std::chrono::system_clock::now();
    vueron::PCDet::scatter();
    auto scatter_endTime = std::chrono::system_clock::now();
    auto scatter_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        scatter_endTime - scatter_startTime);

    auto rpn_startTime = std::chrono::system_clock::now();
    rpn.run(bev_image, rpn_outputs);
    auto rpn_endTime = std::chrono::system_clock::now();
    auto rpn_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        rpn_endTime - rpn_startTime);

    auto post_startTime = std::chrono::system_clock::now();
    vueron::PCDet::postprocess(final_boxes, final_labels, final_scores);
    auto post_endTime = std::chrono::system_clock::now();
    auto post_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        post_endTime - post_startTime);

#ifdef _PROFILE
    std::cout << "Preprocessing: " << pre_sec.count() / 1000.0 << std::endl;
    std::cout << "PFE: " << pfe_sec.count() / 1000.0 << std::endl;
    std::cout << "Scatter: " << scatter_sec.count() / 1000.0 << std::endl;
    std::cout << "RPN: " << rpn_sec.count() / 1000.0 << std::endl;
    std::cout << "Postprocessing: " << post_sec.count() / 1000.0 << std::endl;
#endif
    /*
        Reset buffers
    */
    std::fill(bev_pillar.begin(), bev_pillar.end(), Pillar());
    std::fill(pfe_input.begin(), pfe_input.end(), 0.0f);
    std::fill(pfe_output.begin(), pfe_output.end(), 0.0f);
    std::fill(bev_image.begin(), bev_image.end(), 0.0f);
    std::fill(suppressed.begin(), suppressed.end(), false);
    num_pillars = 0;

    /*
        Clear buffers to have zero length
    */
    voxel_coords.clear();
    voxel_num_points.clear();
    rpn_outputs.clear();
    pre_boxes.clear();
    pre_labels.clear();
    pre_scores.clear();
    post_boxes.clear();
    post_labels.clear();
    post_scores.clear();
};
