#ifndef __PCDET_H_TEST__
#define __PCDET_H_TEST__

#include <cstddef>
#include <memory>  // for unique_ptr
#include <vector>

#include "ort_model.h"
#include "params.h"
#include "post.h"
#include "pre.h"

namespace vueron {
class PCDetCPU {
 private:
  /*
      Buffers for Inference Pipeline
  */
  // preprocess
  std::vector<Pillar> bev_pillar;        // output of voxelization()
  std::vector<size_t> voxel_coords;      // order : (x, y)
  std::vector<size_t> voxel_num_points;  // output of scatter()
  size_t num_pillars;                    // input of scatter()
  std::vector<float> pfe_input;          // input of pfe_run()
  std::vector<float> pfe_output;         // input of scatter()

  // rpn
  std::vector<float> bev_image;  // input of RPN

  // postprocess
  std::vector<std::vector<float>> rpn_outputs;  // output of RPN
  std::vector<BndBox> pre_boxes;                // boxes before NMS
  std::vector<size_t> pre_labels;               // labels before NMS
  std::vector<float> pre_scores;                // scores before NMS
  std::vector<bool> suppressed;                 // mask for nms

  // Ort Models
  std::unique_ptr<OrtModel> pfe;
  std::unique_ptr<OrtModel> rpn;

  /*
      Buffers for Final Predictions
  */
  std::vector<BndBox> post_boxes;   // boxes after NMS
  std::vector<size_t> post_labels;  // labels after NMS
  std::vector<float> post_scores;   // scores after NMS

  void preprocess(const float *points, size_t point_buf_len,
                  size_t point_stride);
  void scatter();
  void postprocess(std::vector<BndBox> &post_boxes,
                   std::vector<size_t> &post_labels,
                   std::vector<float> &post_scores);
  void get_pred(std::vector<PredBox> &boxes) const;

 public:
  PCDetCPU() = delete;
  PCDetCPU(const PCDetCPU &copy) = delete;
  PCDetCPU &operator=(const PCDetCPU &copy) = delete;
  PCDetCPU(const std::string &pfe_path, const std::string &rpn_path);
  ~PCDetCPU() = default;
  void run(const float *points, size_t point_buf_len, size_t point_stride,
           std::vector<PredBox> &boxes);
  void run(const float *points, size_t point_buf_len, size_t point_stride,
           std::vector<BndBox> &final_boxes, std::vector<size_t> &final_labels,
           std::vector<float> &final_scores);
};
}  // namespace vueron

#endif  // __PCDET_H_TEST__
