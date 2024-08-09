#ifndef __PCDET_H__
#define __PCDET_H__

#include <cstddef>
#include <string>
#include <vector>

#include "pcdet-infer-cpu/common/runtimeconfig.h"
#include "pcdet-infer-cpu/ort_model.h"
#include "pcdet-infer-cpu/post.h"
#include "pcdet-infer-cpu/pre.h"

namespace vueron {
class PCDet {
 public:
  std::string version_info;
  virtual void run(const float *points, size_t point_buf_len,
                   size_t point_stride, std::vector<BndBox> &boxes) = 0;
  virtual void run(const float *points, size_t point_buf_len,
                   size_t point_stride, std::vector<Box> &final_boxes,
                   std::vector<size_t> &final_labels,
                   std::vector<float> &final_scores) = 0;
  const std::string &getVersionInfo() const noexcept;
  PCDet() = default;
  PCDet(const PCDet &copy) = delete;
  PCDet &operator=(const PCDet &copy) = delete;
  PCDet(const PCDet &&rhs) = delete;
  PCDet &operator=(const PCDet &&rhs) = delete;
  virtual ~PCDet() = default;
};

class PCDetCPU : public PCDet {
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
  std::vector<Box> pre_boxes;                // boxes before NMS
  std::vector<size_t> pre_labels;               // labels before NMS
  std::vector<float> pre_scores;                // scores before NMS
  std::vector<bool> suppressed;                 // mask for nms

  /*
      Ort Models
  */
  std::unique_ptr<OrtModel> pfe;
  std::unique_ptr<OrtModel> rpn;

  /*
      Buffers for Final Predictions
  */
  std::vector<Box> post_boxes;   // boxes after NMS
  std::vector<size_t> post_labels;  // labels after NMS
  std::vector<float> post_scores;   // scores after NMS

  void preprocess(const float *points, size_t point_buf_len,
                  size_t point_stride);
  void scatter() noexcept;
  void postprocess(std::vector<Box> &post_boxes,
                   std::vector<size_t> &post_labels,
                   std::vector<float> &post_scores);
  void get_pred(std::vector<BndBox> &boxes) const noexcept;

 public:
  PCDetCPU() = delete;
  PCDetCPU(const PCDetCPU &copy) = delete;
  PCDetCPU &operator=(const PCDetCPU &copy) = delete;
  PCDetCPU(const PCDetCPU &&rhs) = delete;
  PCDetCPU &operator=(const PCDetCPU &&rhs) = delete;
  PCDetCPU(const std::string &pfe_path, const std::string &rpn_path,
           const RuntimeConfig *runtimeconfig = nullptr);
  ~PCDetCPU() override;
  void run(const float *points, size_t point_buf_len, size_t point_stride,
           std::vector<BndBox> &boxes) override;
  void run(const float *points, size_t point_buf_len, size_t point_stride,
           std::vector<Box> &final_boxes, std::vector<size_t> &final_labels,
           std::vector<float> &final_scores) override;
};
}  // namespace vueron

#endif  // __PCDET_H__
