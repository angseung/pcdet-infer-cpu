#include "include/pcdet.h"

#include <iostream>

vueron::PCDetCPU::PCDetCPU(const std::string &pfe_path,
                           const std::string &rpn_path)
    : bev_pillar(GRID_Y_SIZE * GRID_X_SIZE, Pillar(MAX_NUM_POINTS_PER_PILLAR)),
      num_pillars(0),
      pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f),
      pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER, 0.0f),
      bev_image(GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER, 0.0f),
      suppressed(NMS_PRE_MAXSIZE, false) {
  std::cout << "PFE Model Initialized with " << PFE_FILE << std::endl;
  std::cout << "RPN Model Initialized with " << RPN_FILE << std::endl;
  std::vector<int64_t> pfe_input_dim{MAX_VOXELS, MAX_NUM_POINTS_PER_PILLAR,
                                     FEATURE_NUM};
  pfe = std::make_unique<OrtModel>(
      pfe_path, pfe_input_dim,
      MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM);

  std::vector<int64_t> rpn_input_dim{1, NUM_FEATURE_SCATTER, GRID_Y_SIZE,
                                     GRID_X_SIZE};
  rpn = std::make_unique<OrtModel>(
      rpn_path, rpn_input_dim, GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER);
};

void vueron::PCDetCPU::preprocess(const float *points,
                                  const size_t point_buf_len,
                                  const size_t point_stride) {
  voxelization(bev_pillar, points, point_buf_len, point_stride);
  num_pillars = point_decoration(bev_pillar, voxel_coords, voxel_num_points,
                                 pfe_input, points, point_stride);
}

void vueron::PCDetCPU::scatter() {
  vueron::scatter(pfe_output, voxel_coords, num_pillars, bev_image);
}

void vueron::PCDetCPU::postprocess(std::vector<BndBox> &post_boxes,
                                   std::vector<size_t> &post_labels,
                                   std::vector<float> &post_scores) {
  decode_to_boxes(rpn_outputs, pre_boxes, pre_labels, pre_scores);
  nms(pre_boxes, pre_scores, suppressed, NMS_THRESH);
  gather_boxes(pre_boxes, pre_labels, pre_scores, post_boxes, post_labels,
               post_scores, suppressed);
}

void vueron::PCDetCPU::get_pred(std::vector<PredBox> &boxes) const {
  for (size_t i = 0; i < post_boxes.size(); i++) {
    PredBox box{};
    box.x = post_boxes[i].x;
    box.y = post_boxes[i].y;
    box.z = post_boxes[i].z;
    box.dx = post_boxes[i].dx;
    box.dy = post_boxes[i].dy;
    box.dz = post_boxes[i].dz;
    box.heading = post_boxes[i].heading;
    box.score = post_scores[i];
    box.label = static_cast<float>(post_labels[i]);

    boxes.push_back(box);
  }
}

void vueron::PCDetCPU::run(const float *points, const size_t point_buf_len,
                           const size_t point_stride,
                           std::vector<PredBox> &boxes) {
  /**
   * @brief
   * It writes predictions into a vector, boxes.
   *
   */
  vueron::PCDetCPU::preprocess(points, point_buf_len, point_stride);
  pfe->run(pfe_input, pfe_output);
  vueron::PCDetCPU::scatter();
  rpn->run(bev_image, rpn_outputs);
  vueron::PCDetCPU::postprocess(post_boxes, post_labels, post_scores);
  vueron::PCDetCPU::get_pred(boxes);

  /*
      Reset buffers
  */
  std::fill(bev_pillar.begin(), bev_pillar.end(),
            Pillar(MAX_NUM_POINTS_PER_PILLAR));
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

void vueron::PCDetCPU::run(const float *points, const size_t point_buf_len,
                           const size_t point_stride,
                           std::vector<BndBox> &final_boxes,
                           std::vector<size_t> &final_labels,
                           std::vector<float> &final_scores) {
  /**
   * @brief
   * It writes predictions into three vectors, final_boxes, final_labels, and
   * final_scores.
   *
   */
  vueron::PCDetCPU::preprocess(points, point_buf_len, point_stride);
  pfe->run(pfe_input, pfe_output);
  vueron::PCDetCPU::scatter();
  rpn->run(bev_image, rpn_outputs);
  vueron::PCDetCPU::postprocess(final_boxes, final_labels, final_scores);

  /*
      Reset buffers
  */
  std::fill(bev_pillar.begin(), bev_pillar.end(),
            Pillar(MAX_NUM_POINTS_PER_PILLAR));
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
