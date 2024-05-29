#ifndef __RPN_H__
#define __RPN_H__

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

void rpn_run(const std::vector<float> &rpn_input,
             std::vector<std::vector<float>> &rpn_output) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, RPN_PATH, session_options);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<int64_t> input_node_dims = {1, NUM_FEATURE_SCATTER, GRID_Y_SIZE,
                                            GRID_X_SIZE};
    size_t input_tensor_size = GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER;

    // create input tensor object from data values
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float *)rpn_input.data(), input_tensor_size,
        input_node_dims.data(), input_node_dims.size());
    assert(input_tensor.IsTensor());

    std::vector<const char *> input_node_names;
    std::vector<const char *> output_node_names;

    size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
#ifdef _DEBUG
        std::cout << "input: " << name.get() << std::endl;
#endif
        input_node_names.push_back(strdup(name.get()));
    }
    assert(input_node_names.size() == 1);

    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
#ifdef _DEBUG
        std::cout << "output: " << name.get() << std::endl;
#endif
        output_node_names.push_back(strdup(name.get()));
    }

    // score model & input tensor, get back output tensor
    auto output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                    &input_tensor, input_node_names.size(),
                    output_node_names.data(), output_node_names.size());
    assert(output_tensors.front().IsTensor());

    // Convert output tensors to std::vector
    rpn_output.reserve(num_output_nodes);

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
        assert(num_elements % (GRID_X_SIZE * GRID_Y_SIZE / 4) == 0);

        // Extract tensor data
        float_array = output_tensors[i].GetTensorMutableData<float>();
        std::vector<float> tensor_data(float_array, float_array + num_elements);
        rpn_output.push_back(tensor_data);
    }
#ifdef _DEBUG
    std::cout << "RPN INFERENCE DONE." << std::endl;
#endif
}
} // namespace vueron

#endif // __RPN_H__
