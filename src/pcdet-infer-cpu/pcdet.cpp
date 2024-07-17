#include "pcdet-infer-cpu/pcdet.h"

#include <iostream>

#include "config.h"
#include "version.h"
#ifdef _PROFILE
#include <chrono>
#endif

const std::string &vueron::PCDet::getVersionInfo() const noexcept {
  return version_info;
}

vueron::PCDetCPU::PCDetCPU(const std::string &pfe_path,
                           const std::string &rpn_path,
                           const RuntimeConfig *runtimeconfig)
    : bev_pillar(GRID_Y_SIZE * GRID_X_SIZE,
                 Pillar{static_cast<size_t>(MAX_NUM_POINTS_PER_PILLAR)}),
      num_pillars(0),
      pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f),
      pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER, 0.0f),
      bev_image(GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER, 0.0f),
      suppressed(NMS_PRE_MAXSIZE, false),
      pfe(std::make_unique<OrtModel>(
          pfe_path,
          std::vector<int64_t>{MAX_VOXELS, MAX_NUM_POINTS_PER_PILLAR,
                               FEATURE_NUM},
          MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM)),
      rpn(std::make_unique<OrtModel>(
          rpn_path,
          std::vector<int64_t>{1, NUM_FEATURE_SCATTER, GRID_Y_SIZE,
                               GRID_X_SIZE},
          GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER)) {
  std::clog << "PFE Model Initialized with " << PFE_FILE << std::endl;
  std::clog << "RPN Model Initialized with " << RPN_FILE << std::endl;
  pre_boxes.reserve(NMS_PRE_MAXSIZE);
  pre_labels.reserve(NMS_PRE_MAXSIZE);
  pre_scores.reserve(NMS_PRE_MAXSIZE);
  post_boxes.reserve(MAX_OBJ_PER_SAMPLE);
  post_labels.reserve(MAX_OBJ_PER_SAMPLE);
  post_scores.reserve(MAX_OBJ_PER_SAMPLE);

  if (runtimeconfig != nullptr) {
    SetRuntimeConfig(*runtimeconfig);
  }
  const std::string git_tag_info{GIT_TAG_VERSION};
  const std::string build_info{BUILD_TIME};

  version_info = "libpcdet-cpu " + git_tag_info + " (" + build_info + ")";
};

vueron::PCDetCPU::~PCDetCPU() = default;

void vueron::PCDetCPU::preprocess(const float *points,
                                  const size_t point_buf_len,
                                  const size_t point_stride) {
  voxelization(bev_pillar, points, point_buf_len, point_stride);
  num_pillars = point_decoration(bev_pillar, voxel_coords, voxel_num_points,
                                 pfe_input, points, point_stride);
}

void vueron::PCDetCPU::scatter() noexcept {
  vueron::scatter(pfe_output, voxel_coords, num_pillars, bev_image);
}

void vueron::PCDetCPU::postprocess(std::vector<vueron::BndBox> &post_boxes,
                                   std::vector<size_t> &post_labels,
                                   std::vector<float> &post_scores) {
  decode_to_boxes(rpn_outputs, pre_boxes, pre_labels, pre_scores);
  nms(pre_boxes, pre_scores, suppressed, NMS_THRESH);
  gather_boxes(pre_boxes, pre_labels, pre_scores, post_boxes, post_labels,
               post_scores, suppressed);
}

void vueron::PCDetCPU::get_pred(std::vector<PredBox> &boxes) const noexcept {
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
#ifdef _PROFILE
  auto pre_startTime = std::chrono::system_clock::now();
#endif
  preprocess(points, point_buf_len, point_stride);
#ifdef _PROFILE
  auto pre_endTime = std::chrono::system_clock::now();
  auto pre_millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
      pre_endTime - pre_startTime);
#endif

#ifdef _PROFILE
  auto pfe_startTime = std::chrono::system_clock::now();
#endif
  pfe->run(pfe_input, pfe_output);
#ifdef _PROFILE
  auto pfe_endTime = std::chrono::system_clock::now();
  auto pfe_millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
      pfe_endTime - pfe_startTime);
#endif

#ifdef _PROFILE
  auto scatter_startTime = std::chrono::system_clock::now();
#endif
  scatter();
#ifdef _PROFILE
  auto scatter_endTime = std::chrono::system_clock::now();
  auto scatter_millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
      scatter_endTime - scatter_startTime);
#endif

#ifdef _PROFILE
  auto rpn_startTime = std::chrono::system_clock::now();
#endif
  rpn->run(bev_image, rpn_outputs);
#ifdef _PROFILE
  auto rpn_endTime = std::chrono::system_clock::now();
  auto rpn_millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
      rpn_endTime - rpn_startTime);
#endif

#ifdef _PROFILE
  auto post_startTime = std::chrono::system_clock::now();
#endif
  postprocess(post_boxes, post_labels, post_scores);
#ifdef _PROFILE
  auto post_endTime = std::chrono::system_clock::now();
  auto post_millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
      post_endTime - post_startTime);
#endif

#ifdef _PROFILE
  auto gather_boxes_startTime = std::chrono::system_clock::now();
#endif
  get_pred(boxes);
#ifdef _PROFILE
  auto gather_boxes_endTime = std::chrono::system_clock::now();
  auto gather_boxes_millisec =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          gather_boxes_endTime - gather_boxes_startTime);
#endif

#ifdef _PROFILE
  std::cout << "Preprocessing: " << pre_millisec.count() / 1000.0 << std::endl;
  std::cout << "PFE: " << pfe_millisec.count() / 1000.0 << std::endl;
  std::cout << "Scatter: " << scatter_millisec.count() / 1000.0 << std::endl;
  std::cout << "RPN: " << rpn_millisec.count() / 1000.0 << std::endl;
  std::cout << "Postprocessing: " << post_millisec.count() / 1000.0
            << std::endl;
  std::cout << "Gather Boxes: " << gather_boxes_millisec.count() / 1000.0
            << std::endl;
#endif

  /*
      Reset buffers
  */
  std::fill(bev_pillar.begin(), bev_pillar.end(),
            Pillar{static_cast<size_t>(MAX_NUM_POINTS_PER_PILLAR)});
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
                           std::vector<vueron::BndBox> &final_boxes,
                           std::vector<size_t> &final_labels,
                           std::vector<float> &final_scores) {
  /**
   * @brief
   * It writes predictions into three vectors, final_boxes, final_labels, and
   * final_scores.
   *
   */
#ifdef _PROFILE
  auto pre_startTime = std::chrono::system_clock::now();
#endif
  preprocess(points, point_buf_len, point_stride);
#ifdef _PROFILE
  auto pre_endTime = std::chrono::system_clock::now();
  auto pre_millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
      pre_endTime - pre_startTime);
#endif

#ifdef _PROFILE
  auto pfe_startTime = std::chrono::system_clock::now();
#endif
  pfe->run(pfe_input, pfe_output);
#ifdef _PROFILE
  auto pfe_endTime = std::chrono::system_clock::now();
  auto pfe_millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
      pfe_endTime - pfe_startTime);
#endif

#ifdef _PROFILE
  auto scatter_startTime = std::chrono::system_clock::now();
#endif
  scatter();
#ifdef _PROFILE
  auto scatter_endTime = std::chrono::system_clock::now();
  auto scatter_millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
      scatter_endTime - scatter_startTime);
#endif

#ifdef _PROFILE
  auto rpn_startTime = std::chrono::system_clock::now();
#endif
  rpn->run(bev_image, rpn_outputs);
#ifdef _PROFILE
  auto rpn_endTime = std::chrono::system_clock::now();
  auto rpn_millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
      rpn_endTime - rpn_startTime);
#endif

#ifdef _PROFILE
  auto post_startTime = std::chrono::system_clock::now();
#endif
  postprocess(final_boxes, final_labels, final_scores);
#ifdef _PROFILE
  auto post_endTime = std::chrono::system_clock::now();
  auto post_millisec = std::chrono::duration_cast<std::chrono::milliseconds>(
      post_endTime - post_startTime);
#endif

#ifdef _PROFILE
  std::cout << "Preprocessing: " << pre_millisec.count() / 1000.0 << std::endl;
  std::cout << "PFE: " << pfe_millisec.count() / 1000.0 << std::endl;
  std::cout << "Scatter: " << scatter_millisec.count() / 1000.0 << std::endl;
  std::cout << "RPN: " << rpn_millisec.count() / 1000.0 << std::endl;
  std::cout << "Postprocessing: " << post_millisec.count() / 1000.0
            << std::endl;
#endif
  /*
      Reset buffers
  */
  std::fill(bev_pillar.begin(), bev_pillar.end(),
            Pillar{static_cast<size_t>(MAX_NUM_POINTS_PER_PILLAR)});
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
