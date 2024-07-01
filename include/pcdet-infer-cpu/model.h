#ifndef __MODEL_H__
#define __MODEL_H__

#include <cstddef>
#include <string>
#include <vector>

#include "type.h"

namespace vueron {

class Model {
 public:
  virtual void run(const std::vector<float> &model_input,
                   std::vector<float> &model_output) = 0;
  virtual void run(const std::vector<float> &model_input,
                   std::vector<std::vector<float>> &model_output) = 0;
  Model() = default;
  Model(const Model &copy) = delete;
  Model &operator=(Model &copy) = delete;
  virtual ~Model() = default;
};

class PCDet {
 public:
  virtual void run(const float *points, const size_t point_buf_len,
                   const size_t point_stride, std::vector<PredBox> &boxes) = 0;
  virtual void run(const float *points, const size_t point_buf_len,
                   const size_t point_stride, std::vector<BndBox> &final_boxes,
                   std::vector<size_t> &final_labels,
                   std::vector<float> &final_scores) = 0;
  virtual std::string getVersionInfo() = 0;
  PCDet() = default;
  PCDet(const PCDet &copy) = delete;
  PCDet &operator=(PCDet &copy) = delete;
  virtual ~PCDet() = default;
};
}  // namespace vueron

#endif
