#ifndef __ORT_MODEL_H__
#define __ORT_MODEL_H__

#include <vector>

#include "model.h"
#include "onnxruntime_cxx_api.h"

namespace vueron {
class OrtModel : public Model {
 private:
  Ort::MemoryInfo memory_info;
  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session session;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;
  std::vector<int64_t> input_node_dims;
  size_t input_tensor_size;
  size_t num_input_nodes;
  size_t num_output_nodes;

 public:
  OrtModel() = delete;
  OrtModel(const std::string &onnx_path,
           const std::vector<int64_t> &input_node_dims,
           size_t input_tensor_size);
  OrtModel(const OrtModel &copy) = delete;
  OrtModel &operator=(const OrtModel &copy) = delete;
  OrtModel(const OrtModel &&rhs) = delete;
  OrtModel &operator=(const OrtModel &&rhs) = delete;
  ~OrtModel() override;

  void run(const std::vector<float> &model_input,
           std::vector<float> &model_output) override;

  void run(const std::vector<float> &model_input,
           std::vector<std::vector<float>> &model_output) override;

  void run(const std::vector<std::vector<float>> &model_input,
           std::vector<std::vector<float>> &model_output) override;
};
}  // namespace vueron

#endif  //__ORT_MODEL_H__
