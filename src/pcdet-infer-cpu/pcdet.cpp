#include "pcdet-infer-cpu/pcdet.h"

vueron::PCDet::PCDet()
    : bev_pillar(GRID_Y_SIZE * GRID_X_SIZE),
      pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f),
      pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER, 0.0f),
      bev_image(GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER, 0.0f),
      pre_boxes(MAX_BOX_NUM_BEFORE_NMS, {0.0f}),
      pre_labels(MAX_BOX_NUM_AFTER_NMS, 0),
      pre_scores(MAX_BOX_NUM_AFTER_NMS, 0.0f) {}

vueron::PCDet::~PCDet(){};

void vueron::PCDet::preprocess() {}

void vueron::PCDet::pfe_run() { int a = 1; }

void vueron::PCDet::rpn_run() { int a = 1; }

void vueron::PCDet::postprocess() { int a = 1; }

std::vector<vueron::PredBox> &vueron::PCDet::get_pred() {
    std::vector<PredBox> boxes;
    return boxes;
}
