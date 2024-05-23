#ifndef RPN_H
#define RPN_H

#include "config.h"
#include "onnxruntime_cxx_api.h"
#include "params.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory.h>
#include <numeric>
#include <random>
#include <vector>

namespace vueron {

void rpn_run(const std::vector<float> &rpn_input
             //  std::vector<float> &rpn_output
) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, RPN_PATH, session_options);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<int64_t> input_node_dims = {GRID_Y_SIZE, GRID_X_SIZE,
                                            RPN_INPUT_NUM_CHANNELS};
    size_t input_tensor_size =
        GRID_Y_SIZE * GRID_X_SIZE * RPN_INPUT_NUM_CHANNELS;

    // create input tensor object from data values
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float *)rpn_input.data(), input_tensor_size,
        input_node_dims.data(), 3);
    assert(input_tensor.IsTensor());

    std::vector<const char *> input_node_names;
    std::vector<const char *> output_node_names;

    size_t num_input_nodes = session.GetInputCount();
    // std::vector<std::string> inputNames;
    for (size_t i = 0; i < num_input_nodes; ++i) {
        Ort::AllocatedStringPtr name =
            session.GetInputNameAllocated(i, allocator);
        std::cout << "input: " << name << std::endl;
        input_node_names.push_back(name.get());
    }

    size_t num_output_nodes = session.GetOutputCount();
    // std::vector<std::string> outputNames;
    for (size_t i = 0; i < num_output_nodes; ++i) {
        Ort::AllocatedStringPtr name =
            session.GetOutputNameAllocated(i, allocator);
        std::cout << "output: " << name << std::endl;
        output_node_names.push_back(name.get());
    }

    // score model & input tensor, get back output tensor
    auto output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                    &input_tensor, input_node_names.size(),
                    output_node_names.data(), output_node_names.size());
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    // Get pointer to output tensor float values
    // Get the first (and assumed to be only) output tensor
    Ort::Value &output_tensor = output_tensors.front();

    // Get the shape of the output tensor
    auto output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    auto output_dims_count = output_tensor_info.GetDimensionsCount();
    size_t output_size = output_tensor_info.GetElementCount();
    float *floatarr = output_tensor.GetTensorMutableData<float>();

    assert(output_size == MAX_VOXELS * RPN_INPUT_NUM_CHANNELS);
    assert(output_dims.size() == output_dims_count && output_dims_count == 2);
    assert(output_dims[0] == MAX_VOXELS);
    assert(output_dims[1] == RPN_INPUT_NUM_CHANNELS);

    // Resize the output vector to fit the output tensor data
    // rpn_output.resize(output_size);

    // Copy the output tensor data to the output vector
    // std::copy(floatarr, floatarr + output_size, rpn_output.begin());
#ifdef _DEBUG
    std::cout << "INFERENCE DONE." << std::endl;
#endif
}
} // namespace vueron

#endif