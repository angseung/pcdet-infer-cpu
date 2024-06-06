#ifndef __PCDET_H__
#define __PCDET_H__

#include <cstddef>
#include <memory>  // for unique_ptr
#include <vector>

#include "pcdet-infer-cpu/common/runtimeconfig.h"
#include "pcdet-infer-cpu/ort_model.h"
#include "pcdet-infer-cpu/post.h"
#include "pcdet-infer-cpu/pre.h"

namespace vueron {
class PCDet {
 private:
  /*
      Buffers for Inference Pipeline
  */
  std::vector<Pillar> bev_pillar;               // output of voxelization()
  std::vector<size_t> voxel_coords;             // order : (x, y)
  std::vector<size_t> voxel_num_points;         // output of scatter()
  size_t num_pillars;                           // input of scatter()
  std::vector<float> pfe_input;                 // input of pfe_run()
  std::vector<float> pfe_output;                // input of scatter()
  std::vector<float> bev_image;                 // input of RPN
  std::vector<std::vector<float>> rpn_outputs;  // output of RPN
  std::vector<BndBox> pre_boxes;                // boxes before NMS
  std::vector<size_t> pre_labels;               // labels before NMS
  std::vector<float> pre_scores;                // scores before NMS
  std::vector<bool> suppressed;                 // mask for nms

  std::string pfe_path;
  std::vector<int64_t> pfe_input_dim;
  std::unique_ptr<OrtModel> pfe;

  std::string rpn_path;
  std::vector<int64_t> rpn_input_dim;
  std::unique_ptr<OrtModel> rpn;
  RuntimeConfig *runtimeconfig;

  /*
      Buffers for Final Predictions
  */
  std::vector<BndBox> post_boxes;   // boxes after NMS
  std::vector<size_t> post_labels;  // labels after NMS
  std::vector<float> post_scores;   // scores after NMS

  void preprocess(const float *points, const size_t &point_buf_len,
                  const size_t &point_stride);
  void scatter();
  void postprocess(std::vector<vueron::BndBox> &post_boxes,
                   std::vector<size_t> &post_labels,
                   std::vector<float> &post_scores);
  void get_pred(std::vector<PredBox> &boxes);

 public:
  PCDet();
  PCDet(PCDet &copy) = delete;
  PCDet(const std::string &pfe_path, const std::string &rpn_path,
        RuntimeConfig *runtimeconfig = nullptr);
  ~PCDet();
  void do_infer(const float *points, const size_t &point_buf_len,
                const size_t &point_stride, std::vector<PredBox> &boxes);
  void do_infer(const float *points, const size_t &point_buf_len,
                const size_t &point_stride,
                std::vector<vueron::BndBox> &final_boxes,
                std::vector<size_t> &final_labels,
                std::vector<float> &final_scores);
};
}  // namespace vueron

#endif  // __PCDET_H__
