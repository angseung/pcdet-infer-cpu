#include "pcdet-infer-cpu/pcdet_c.h"

#include <cassert>
#include <memory>

#include "pcdet-infer-cpu/pcdet.h"

extern "C" {

static std::unique_ptr<vueron::PCDetCPU> pcdet;
static std::vector<vueron::BndBox> g_nms_pred;
static std::vector<float> g_nms_score;
static std::vector<size_t> g_nms_labels;
static std::string version;

const char* get_pcdet_cpu_version(void) {
  version = std::string{pcdet->version_info};

  return version.c_str();
};

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

size_t pcdet_run(const float* points, const int point_buf_len,
                 const int point_stride, float** score, size_t** label,
                 Box** box) {
  // check point input size
  assert(point_buf_len % point_stride == 0);

  // run inference session
  pcdet->run(points, point_buf_len, point_stride, g_nms_pred, g_nms_labels,
             g_nms_score);

  // check predicted buffer size
  assert(g_nms_pred.size() == g_nms_labels.size() &&
         g_nms_labels.size() == g_nms_score.size());

  // copy address of output buffer to return pointers
  size_t num_preds = g_nms_labels.size();
  *score = g_nms_score.data();
  *label = g_nms_labels.data();
  *box = (Box*)g_nms_pred.data();

  // clear global static buffers
  g_nms_score.clear();
  g_nms_pred.clear();
  g_nms_labels.clear();

  return num_preds;
}

void pcdet_finalize(void) {
  // destruct pcdet model
  pcdet = nullptr;

  // clear global static buffers
  g_nms_score.clear();
  g_nms_pred.clear();
  g_nms_labels.clear();
}

}  // extern "C"