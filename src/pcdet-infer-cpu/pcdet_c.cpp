#include "pcdet-infer-cpu/pcdet_c.h"

#include <cassert>
#include <memory>

#include "pcdet-infer-cpu/pcdet.h"

extern "C" {

static std::unique_ptr<vueron::PCDetCPU> pcdet = nullptr;
static std::vector<BndBox> g_nms_boxes;

// Temporary global static buffers for pcdet->pcdet_run()
static std::vector<Box> g_nms_pred;
static std::vector<float> g_nms_score;
static std::vector<size_t> g_nms_labels;

const char* get_pcdet_cpu_version(void) {
  static std::string version{};
  version = std::string{pcdet->version_info};

  return version.c_str();
}

void pcdet_initialize(const char* metadata_path,
                      const struct RuntimeConfig* runtimeconfig) {
  const std::string metadata_path_string{metadata_path};

  // MAX_OBJ_PER_SAMPLE is maximum size of each vectors.
  g_nms_boxes.reserve(MAX_OBJ_PER_SAMPLE);
  g_nms_pred.reserve(MAX_OBJ_PER_SAMPLE);
  g_nms_score.reserve(MAX_OBJ_PER_SAMPLE);
  g_nms_labels.reserve(MAX_OBJ_PER_SAMPLE);

  // initialize model with metadata
  vueron::LoadMetadata(metadata_path_string);
  pcdet = std::make_unique<vueron::PCDetCPU>(PFE_FILE, RPN_FILE, runtimeconfig);

  // logging Metadata & RuntimeConfig
  std::cout << vueron::GetMetadata() << std::endl;
  std::cout << *runtimeconfig << std::endl;
}

int pcdet_run(const float* points, const int point_buf_len,
              const int point_stride, BndBox** box) {
  // check point input size
  assert(point_buf_len % point_stride == 0);
  g_nms_boxes.clear();
  assert(g_nms_boxes.empty());

  // run inference session
  pcdet->run(points, point_buf_len, point_stride, g_nms_pred, g_nms_labels,
             g_nms_score);

  // check predicted buffer size
  assert(g_nms_pred.size() == g_nms_labels.size() &&
         g_nms_labels.size() == g_nms_score.size());

  // copy address of output buffer to return pointers
  const int num_preds = static_cast<int>(g_nms_labels.size());
  for (int i = 0; i < num_preds; i++) {
    BndBox temp_box{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    temp_box.x = g_nms_pred[i].x;
    temp_box.y = g_nms_pred[i].y;
    temp_box.z = g_nms_pred[i].z;
    temp_box.dx = g_nms_pred[i].dx;
    temp_box.dy = g_nms_pred[i].dy;
    temp_box.dz = g_nms_pred[i].dz;
    temp_box.heading = g_nms_pred[i].heading;
    temp_box.label = static_cast<float>(g_nms_labels[i]);
    temp_box.score = g_nms_score[i];

    // append temp_box into g_nms_boxes
    g_nms_boxes.push_back(temp_box);
  }

  // clear global static buffers
  g_nms_score.clear();
  g_nms_pred.clear();
  g_nms_labels.clear();
  *box = g_nms_boxes.data();

  return num_preds;
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
