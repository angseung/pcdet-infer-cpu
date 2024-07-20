#include "pcdet-infer-cpu/pcdet_c.h"

#include <cassert>
#include <memory>

#include "pcdet-infer-cpu/pcdet.h"

extern "C" {

static std::unique_ptr<vueron::PCDetCPU> pcdet;
static std::vector<vueron::BndBox> g_nms_pred;
static std::vector<float> g_nms_score;
static std::vector<size_t> g_nms_labels;

const char* get_pcdet_cpu_version(void) {
  static std::string version{pcdet->version_info};

  return version.c_str();
};

void pcdet_initialize(const char* metadata_path,
                      const struct RuntimeConfig* runtimeconfig) {
  const std::string metadata_path_string{metadata_path};
  vueron::LoadMetadata(metadata_path_string);
  pcdet = std::make_unique<vueron::PCDetCPU>(PFE_FILE, RPN_FILE, runtimeconfig);
  std::cout << vueron::GetMetadata() << std::endl;
  std::cout << *runtimeconfig << std::endl;
}

size_t pcdet_run(const float* points, const int point_buf_len,
                 const int point_stride, float** score, size_t** label,
                 Box** box) {
  assert(point_buf_len % point_stride == 0);
  pcdet->run(points, point_buf_len, point_stride, g_nms_pred, g_nms_labels,
             g_nms_score);

  assert(g_nms_pred.size() == g_nms_labels.size() &&
         g_nms_labels.size() == g_nms_score.size());

  size_t num_preds = g_nms_labels.size();
  *score = g_nms_score.data();
  *label = g_nms_labels.data();
  *box = (Box*)g_nms_pred.data();

  g_nms_score.clear();
  g_nms_pred.clear();
  g_nms_labels.clear();

  return num_preds;
}

void pcdet_finalize(void) {
  pcdet = nullptr;
  g_nms_score.clear();
  g_nms_pred.clear();
  g_nms_labels.clear();
}

}  // extern "C"
