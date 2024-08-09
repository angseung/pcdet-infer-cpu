#include "pcdet-infer-cpu/pcdet_c.h"

#include <cassert>
#include <memory>

#include "pcdet-infer-cpu/pcdet.h"

extern "C" {

static std::unique_ptr<vueron::PCDetCPU> pcdet;
static std::vector<Box> g_nms_pred;
static std::vector<float> g_nms_score;
static std::vector<size_t> g_nms_labels;
static std::vector<BndBox> g_nms_boxes;

const char* get_pcdet_cpu_version(void) {
  static std::string version{};
  version = std::string{pcdet->version_info};

  return version.c_str();
}

void pcdet_initialize(const char* metadata_path,
                      const struct RuntimeConfig* runtimeconfig) {
  const std::string metadata_path_string{metadata_path};

  // initialize model with metadata
  vueron::LoadMetadata(metadata_path_string);
  pcdet = std::make_unique<vueron::PCDetCPU>(PFE_FILE, RPN_FILE, runtimeconfig);

  // logging configurations
  std::cout << vueron::GetMetadata() << std::endl;
  std::cout << *runtimeconfig << std::endl;
}

int pcdet_run(const float* points, const int point_buf_len,
              const int point_stride, BndBox** box) {
  // check point input size
  assert(point_buf_len % point_stride == 0);
  g_nms_boxes.clear();

  // run inference session
  pcdet->run(points, point_buf_len, point_stride, g_nms_pred, g_nms_labels,
             g_nms_score);

  // check predicted buffer size
  assert(g_nms_pred.size() == g_nms_labels.size() &&
         g_nms_labels.size() == g_nms_score.size());
  assert(g_nms_boxes.empty());

  // copy address of output buffer to return pointers
  const int num_preds = static_cast<int>(g_nms_labels.size());
  for (size_t i = 0; i < num_preds; i++) {
    BndBox box{};
    box.x = g_nms_pred[i].x;
    box.y = g_nms_pred[i].y;
    box.z = g_nms_pred[i].z;
    box.dx = g_nms_pred[i].dx;
    box.dy = g_nms_pred[i].dy;
    box.dz = g_nms_pred[i].dz;
    box.heading = g_nms_pred[i].heading;
    box.label = static_cast<float>(g_nms_labels[i]);
    box.score = g_nms_score[i];
    g_nms_boxes.push_back(box);
  }

  // clear global static buffers
  g_nms_score.clear();
  g_nms_pred.clear();
  g_nms_labels.clear();
  *box = g_nms_boxes.data();

  return num_preds;
}

void pcdet_finalize(void) {
  // destruct pcdet model
  pcdet = nullptr;

  // clear global static buffers
  g_nms_score.clear();
  g_nms_pred.clear();
  g_nms_labels.clear();
  g_nms_boxes.clear();
}

}  // extern "C"
