#include "pcdet-infer-cpu/pcdet.h"
#include "pcdet-infer-cpu/post.h"
#include "pcdet-infer-cpu/pre.h"
#include "pcdet-infer-cpu/rpn.h"

vueron::PCDet::PCDet()
    : bev_pillar(GRID_Y_SIZE * GRID_X_SIZE),
      pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f),
      pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER, 0.0f),
      bev_image(GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER, 0.0f),
      suppressed(MAX_BOX_NUM_BEFORE_NMS, false),
      num_pillars(0){
          // TODO: Implement Onnxruntime Model Class
      };

vueron::PCDet::~PCDet(){};

void vueron::PCDet::preprocess(const float *points, const size_t point_buf_len,
                               const size_t point_stride) {
    vueron::voxelization(bev_pillar, points, point_buf_len, point_stride);
    num_pillars =
        vueron::point_decoration(bev_pillar, voxel_coords, voxel_num_points,
                                 pfe_input, points, point_stride);
}

void vueron::PCDet::pfe_run(void) { vueron::pfe_run(pfe_input, pfe_output); }

void vueron::PCDet::scatter(void) {
    vueron::scatter(pfe_output, voxel_coords, num_pillars, bev_image);
}

void vueron::PCDet::rpn_run(void) { vueron::rpn_run(bev_image, rpn_outputs); }

void vueron::PCDet::postprocess(std::vector<vueron::BndBox> &final_boxes,
                                std::vector<size_t> &final_labels,
                                std::vector<float> &final_scores) {
    vueron::decode_to_boxes(rpn_outputs, pre_boxes, pre_labels, pre_scores);
    vueron::nms(pre_boxes, pre_scores, suppressed, IOU_THRESH);
    vueron::gather_boxes(pre_boxes, pre_scores, pre_labels, final_boxes,
                         final_scores, final_labels, suppressed);
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
    vueron::PCDet::preprocess(points, point_buf_len, point_stride);
    vueron::PCDet::pfe_run();
    vueron::PCDet::scatter();
    vueron::PCDet::rpn_run();
    vueron::PCDet::postprocess(post_boxes, post_labels, post_scores);
    vueron::PCDet::get_pred(final_boxes);

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
    vueron::PCDet::preprocess(points, point_buf_len, point_stride);
    vueron::PCDet::pfe_run();
    vueron::PCDet::scatter();
    vueron::PCDet::rpn_run();
    vueron::PCDet::postprocess(final_boxes, final_labels, final_scores);
    /**
     * @brief
     * It writes predictions into three vectors, final_boxes, final_labels, and
     * final_scores.
     *
     */

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
