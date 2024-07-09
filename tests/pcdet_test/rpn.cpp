#include "include/rpn.h"

#include <memory.h>

#include <cassert>

#include "include/params.h"
#include "onnxruntime_cxx_api.h"

void vueron::rpn_run(const std::vector<float> &rpn_input,
                     std::vector<std::vector<float>> &rpn_output) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;
  Ort::Session session(env, RPN_FILE, session_options);
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::AllocatorWithDefaultOptions allocator;

  const std::vector<int64_t> input_node_dims = {1, NUM_FEATURE_SCATTER,
                                                GRID_Y_SIZE, GRID_X_SIZE};
  const size_t input_tensor_size =
      GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER;

  // create input tensor object from data values
  const auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;

  const size_t num_input_nodes = session.GetInputCount();
  for (size_t i = 0; i < num_input_nodes; ++i) {
    auto name = session.GetInputNameAllocated(i, allocator);
    input_node_names.push_back(strdup(name.get()));
  }
  assert(input_node_names.size() == 1);

  const size_t num_output_nodes = session.GetOutputCount();
  for (size_t i = 0; i < num_output_nodes; ++i) {
    auto name = session.GetOutputNameAllocated(i, allocator);
    output_node_names.push_back(strdup(name.get()));
  }

  // Make input tensor
  const auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(rpn_input.data()), input_tensor_size,
      input_node_dims.data(), input_node_dims.size());
  assert(input_tensor.IsTensor());

  // Run ort session
  auto output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                  &input_tensor, input_node_names.size(),
                  output_node_names.data(), output_node_names.size());
  assert(output_tensors.front().IsTensor());

  // Convert output tensors into std::vector
  rpn_output.reserve(num_output_nodes);

  for (size_t i = 0; i < num_output_nodes; ++i) {
    // Get tensor shape and size
    auto type_info = output_tensors[i].GetTensorTypeAndShapeInfo();
    auto tensor_shape = type_info.GetShape();
    size_t num_elements = 1;

    // Get tensor data size
    for (const auto dim : tensor_shape) {
      num_elements *= dim;
    }
    assert(num_elements % (GRID_X_SIZE * GRID_Y_SIZE / 4) == 0);

    // Extract tensor data
    auto *float_array = output_tensors[i].GetTensorMutableData<float>();
    std::vector<float> tensor_data(float_array, float_array + num_elements);
    rpn_output.push_back(tensor_data);
  }
}
