#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>

namespace vueron {

class Model {
 public:
  virtual void run(const std::vector<float> &model_input,
                   std::vector<float> &model_output) = 0;
  virtual void run(const std::vector<float> &model_input,
                   std::vector<std::vector<float>> &model_output) = 0;
  Model() = default;
  virtual ~Model() = default;
};

class PCDetModel {};
}  // namespace vueron

#endif
