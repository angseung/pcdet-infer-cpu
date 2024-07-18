#include "pcdet_c.h"

#include <memory>

#include "pcdet-infer-cpu/pcdet.h"

static std::unique_ptr<vueron::PCDetCPU> pcdet;
static std::vector<vueron::BndBox> nms_pred;
static std::vector<float> nms_score;
static std::vector<size_t> nms_labels;

void pcdet_initialize(const char* metadata_path,
                      const RuntimeConfig* runtimeconfig) {
  const std::string metadata_path_string{metadata_path};
  vueron::LoadMetadata(metadata_path_string);
  pcdet = std::make_unique<vueron::PCDetCPU>(PFE_FILE, RPN_FILE, runtimeconfig);
  std::cout << vueron::GetMetadata() << std::endl;
  std::cout << *runtimeconfig << std::endl;
}

size_t pcdet_run(const float* points, const int points_num,
                 const int point_stride, float** score, size_t** label,
                 Box** box) {
  static size_t num_preds;
  pcdet->run(points, points_num, point_stride, nms_pred, nms_labels, nms_score);

  num_preds = nms_labels.size();
  *score = nms_score.data();
  *label = nms_labels.data();
  *box = (Box*)nms_pred.data();

  nms_score.clear();
  nms_pred.clear();
  nms_pred.clear();

  return num_preds;
}

void pcdet_finalize() { pcdet = nullptr; }
