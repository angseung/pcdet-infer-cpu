#ifndef __MODEL_H__
#define __MODEL_H__

#include <cstddef>
#include <string>
#include <vector>

#include "pcdet-infer-cpu/common/box.h"

namespace vueron {

class Model {
 public:
  virtual void run(const std::vector<float> &model_input,
                   std::vector<float> &model_output) = 0;
  virtual void run(const std::vector<float> &model_input,
                   std::vector<std::vector<float>> &model_output) = 0;
  virtual void run(const std::vector<std::vector<float>> &model_input,
                   std::vector<std::vector<float>> &model_output) = 0;
  Model() = default;
  Model(const Model &copy) = delete;
  Model &operator=(const Model &copy) = delete;
  Model(const Model &&rhs) = delete;
  Model &operator=(const Model &&rhs) = delete;
  virtual ~Model() = default;
};
}  // namespace vueron

#endif
