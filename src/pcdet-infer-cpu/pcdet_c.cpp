#include "pcdet-infer-cpu/pcdet_c.h"

#include <cassert>
#include <memory>

#include "pcdet-infer-cpu/pcdet.h"

extern "C" {

static std::unique_ptr<vueron::PCDetCPU> pcdet;
static std::vector<Bndbox> g_nms_boxes;

// Global static buffers for pcdet->pcdet_run()
static std::vector<Box> g_nms_pred;
static std::vector<float> g_nms_score;
static std::vector<size_t> g_nms_labels;

const char* GetlibDLVersion(void) {
  static std::string version;
  version = std::string{pcdet->version_info};

  return version.c_str();
}

const char* GetCUDATRTVersion(void) {
  static std::string trt_version_info;
  std::string cuda_version{"None"};
  std::string trt_version{"None"};

  trt_version_info = "cuda_" + cuda_version + "_trt_" + trt_version;

  return trt_version_info.c_str();
}

void pcdet_initialize(const char* metadata_path, const char* onnx_hash,
                      const struct RuntimeConfig* runtimeconfig) {
  // Use "struct" keyword for compatibility with C.
  const std::string metadata_path_string{metadata_path};
  std::ignore = onnx_hash;  // unused param for same interface with pcdet-infer.

  // initialize model with metadata
  vueron::LoadMetadata(metadata_path_string);
  pcdet = std::make_unique<vueron::PCDetCPU>(PFE_FILE, RPN_FILE, runtimeconfig);

  // MAX_OBJ_PER_SAMPLE is the maximum number of each vector.
  g_nms_boxes.reserve(MAX_OBJ_PER_SAMPLE);
  g_nms_pred.reserve(MAX_OBJ_PER_SAMPLE);
  g_nms_score.reserve(MAX_OBJ_PER_SAMPLE);
  g_nms_labels.reserve(MAX_OBJ_PER_SAMPLE);

  // logging Metadata & RuntimeConfig
  std::cout << vueron::GetMetadata() << std::endl;
  std::cout << vueron::GetRuntimeConfig() << std::endl;

  // logging version info
  std::cout << std::string{GetlibDLVersion()} << std::endl;
}

int pcdet_infer(size_t points_size, const float* points,
                struct Bndbox** boxes) {
  const auto _ = pcdet_infer(points_size, points);
  *boxes = g_nms_boxes.data();

  return static_cast<int>(g_nms_labels.size());
}

void pcdet_finalize(void) {
  // destruct pcdet model, which is std::unique_ptr
  pcdet = nullptr;

  // reset global static buffers
  g_nms_score.clear();
  g_nms_pred.clear();
  g_nms_labels.clear();
  g_nms_boxes.clear();

  // check vector is empty
  assert(g_nms_score.empty() && g_nms_pred.empty() && g_nms_labels.empty() &&
         g_nms_boxes.empty());
}

}  // extern "C"

std::vector<Bndbox> pcdet_infer(size_t points_size, const float* points) {
  // clear global static buffers
  g_nms_score.clear();
  g_nms_pred.clear();
  g_nms_labels.clear();

  // check point input size
  g_nms_boxes.clear();
  assert(g_nms_boxes.empty());

  // run inference session
  pcdet->run(points, points_size, 4, g_nms_pred, g_nms_labels, g_nms_score);

  // check predicted buffer size
  assert(g_nms_pred.size() == g_nms_labels.size() &&
         g_nms_labels.size() == g_nms_score.size());

  // copy address of output buffer to return pointers
  const int n_pred_boxes = static_cast<int>(g_nms_labels.size());
  for (int i = 0; i < n_pred_boxes; i++) {
    Bndbox temp_box{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0.0f};
    temp_box.x = g_nms_pred[i].x;
    temp_box.y = g_nms_pred[i].y;
    temp_box.z = g_nms_pred[i].z;
    temp_box.dx = g_nms_pred[i].dx;
    temp_box.dy = g_nms_pred[i].dy;
    temp_box.dz = g_nms_pred[i].dz;
    temp_box.heading = g_nms_pred[i].heading;
    temp_box.score = g_nms_score[i];
    temp_box.label = static_cast<int>(g_nms_labels[i]);

    // append temp_box into g_nms_boxes
    g_nms_boxes.push_back(temp_box);
  }

  return g_nms_boxes;
}
