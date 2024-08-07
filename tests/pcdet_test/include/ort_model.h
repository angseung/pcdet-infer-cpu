#ifndef __ORT_MODEL_H_TEST__
#define __ORT_MODEL_H_TEST__

#include <vector>

#include "onnxruntime_cxx_api.h"

namespace vueron {
class OrtModel {
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
  ~OrtModel() = default;

  void run(const std::vector<float> &model_input,
           std::vector<float> &model_output);

  void run(const std::vector<float> &model_input,
           std::vector<std::vector<float>> &model_output);
};
}  // namespace vueron

#endif  //__ORT_MODEL_H_TEST__
