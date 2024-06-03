#include "pcdet-infer-cpu/ort_model.h"
#include <cassert>

vueron::OrtModel::OrtModel(const std::string onnx_path,
                           const std::vector<int64_t> input_node_dims,
                           const size_t input_tensor_size)
    : memory_info(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      env(ORT_LOGGING_LEVEL_WARNING, "test"),
      session(env, onnx_path.c_str(), session_options),
      input_node_dims(input_node_dims), input_tensor_size(input_tensor_size) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(strdup(name.get()));
    }

    num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(strdup(name.get()));
    }
}

vueron::OrtModel::~OrtModel(){};

void vueron::OrtModel::run(const std::vector<float> &model_input,
                           std::vector<std::vector<float>> &model_output) {
    /*
        This function supports only single input & multiple output models
    */
    assert(input_node_names.size() == 1);
    assert(output_node_names.size() != 1);

    // make input tensor
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float *)model_input.data(), input_tensor_size,
        input_node_dims.data(), input_node_dims.size());
    assert(input_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    auto output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                    &input_tensor, input_node_names.size(),
                    output_node_names.data(), output_node_names.size());
    assert(output_tensors.front().IsTensor());

    // Convert output tensors to std::vector
    model_output.reserve(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; ++i) {
        float *float_array;
        size_t num_elements;

        // Get tensor shape and size
        auto type_info = output_tensors[i].GetTensorTypeAndShapeInfo();
        auto tensor_shape = type_info.GetShape();
        num_elements = 1;

        // Get tensor data size
        for (auto dim : tensor_shape) {
            num_elements *= dim;
        }

        // Extract tensor data
        float_array = output_tensors[i].GetTensorMutableData<float>();
        std::vector<float> tensor_data(float_array, float_array + num_elements);
        model_output.push_back(tensor_data);
    }
};

void vueron::OrtModel::run(const std::vector<float> &model_input,
                           std::vector<float> &model_output) {
    /*
        This function supports only single input & single output models
    */
    assert(input_node_names.size() == 1);
    assert(output_node_names.size() == 1);

    // make input tensor
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float *)model_input.data(), input_tensor_size,
        input_node_dims.data(), input_node_dims.size());
    assert(input_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1,
        output_node_names.data(), output_node_names.size());
    assert(output_tensors.front().IsTensor());

    // Get pointer to output tensor float values
    // Get the first (and assumed to be only) output tensor
    Ort::Value &output_tensor = output_tensors.front();

    // Get the shape of the output tensor
    auto output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    auto output_dims_count = output_tensor_info.GetDimensionsCount();
    size_t output_size = output_tensor_info.GetElementCount();
    float *floatarr = output_tensor.GetTensorMutableData<float>();

    // Resize the output vector to fit the output tensor data
    model_output.resize(output_size);

    // Copy the output tensor data to the output vector
    std::copy(floatarr, floatarr + output_size, model_output.begin());
};
