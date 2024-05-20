#ifndef PRE_H
#define PRE_H

#include "config.h"
#include "onnxruntime_cxx_api.h"
#include "params.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory.h>
#include <numeric> // std::iota
#include <random>
#include <vector>

namespace vueron {

struct BaseVoxel {
    float x = 0.0f, y = 0.0f, z = 0.0f;
#if NUM_POINT_VALUES >= 4
    float w = 0.0f;
#endif
};

struct Voxel {
    float x = 0, y = 0, z = 0;
#if NUM_POINT_VALUES >= 4
    float w = 0;
#endif
    float offset_from_mean_x = 0, offset_from_mean_y = 0,
          offset_from_mean_z = 0;
    float offset_from_center_x = 0, offset_from_center_y = 0,
          offset_from_center_z = 0;
    size_t grid_x = 0;
    size_t grid_y = 0;
    bool is_valid = false;
};

struct Pillar {
    size_t point_index[MAX_NUM_POINTS_PER_PILLAR] = {0};
    size_t pillar_grid_x = 0;
    size_t pillar_grid_y = 0;
    size_t point_num_in_pillar = 0;
    bool is_empty = true;
};

void voxelization(std::vector<Pillar> &bev_pillar, const float *points,
                  size_t points_buf_len, size_t point_stride) {
    // check grid size
    assert(GRID_X_SIZE == (float)((MAX_X_RANGE - MIN_X_RANGE) / VOXEL_X_SIZE));
    assert(GRID_Y_SIZE == (float)((MAX_Y_RANGE - MIN_Y_RANGE) / VOXEL_Y_SIZE));
    assert(1 == (float)((MAX_Z_RANGE - MIN_Z_RANGE) / VOXEL_Z_SIZE));

    // check buffer size
    assert(points_buf_len % point_stride == 0);

    std::mt19937 rng(RANDOM_SEED);

    size_t points_num = points_buf_len / point_stride;
    std::vector<size_t> indices(points_num, 0);
    std::vector<BaseVoxel> shuffled(points_num);
    std::iota(indices.begin(), indices.end(), 0);
    size_t processed = 0;

// TODO: Test with _SHIFFLE ON
#if _SHIFFLE == ON
    std::shuffle(indices.begin(), indices.end(), rng);
#endif

    for (size_t i : indices) {
        // check max_point_num_per_frame
        if (processed > MAX_POINTS_NUM)
            // requires valiation here
            break;
        float point_x = points[point_stride * i];
        float point_y = points[point_stride * i + 1];
        float point_z = points[point_stride * i + 2];

#if NUM_POINT_VALUES >= 4
        float point_i = points[point_stride * i + 3];
#endif
        processed++;
        assert(processed < MAX_POINTS_NUM);

        size_t voxel_id_x = floorf((point_x - MIN_X_RANGE) / VOXEL_X_SIZE);
        size_t voxel_id_y = floorf((point_y - MIN_Y_RANGE) / VOXEL_Y_SIZE);

        // skip if out-of-range point
        if (point_x < MIN_X_RANGE || point_x > MAX_X_RANGE ||
            point_y < MIN_Y_RANGE || point_y > MAX_Y_RANGE ||
            point_z < MIN_Z_RANGE || point_z > MAX_Z_RANGE) {
            continue;
        }

        // check out-of-range point
        assert(point_x >= MIN_X_RANGE && point_x <= MAX_X_RANGE &&
               point_y >= MIN_Y_RANGE && point_y <= MAX_Y_RANGE &&
               point_z >= MIN_Z_RANGE && point_z <= MAX_Z_RANGE);

        // check out-of-range grid
        assert(voxel_id_x < GRID_X_SIZE && voxel_id_y < GRID_Y_SIZE);

        size_t voxel_index = voxel_id_y * GRID_X_SIZE + voxel_id_x;

        // bev_pillar : GRID_Y_SIZE * GRID_X_SIZE vector<Pillar>
        if (bev_pillar[voxel_index].point_num_in_pillar < 20) {
            size_t voxel_index_in_pillar =
                bev_pillar[voxel_index].point_num_in_pillar;
            bev_pillar[voxel_index].pillar_grid_x = voxel_id_x;
            bev_pillar[voxel_index].pillar_grid_y = voxel_id_y;
            bev_pillar[voxel_index].point_index[voxel_index_in_pillar] = i;
            bev_pillar[voxel_index].point_num_in_pillar++;
            assert(bev_pillar[voxel_index].point_num_in_pillar <=
                   MAX_NUM_POINTS_PER_PILLAR);
            bev_pillar[voxel_index].is_empty = false;
        }
    }

#ifdef _DEBUG
    size_t count_voxel = 0;
    for (Pillar pillar : bev_pillar) {
        if (!pillar.is_empty) {
            count_voxel++;
        }
    }
    std::cout << "Number of Voxel: " << count_voxel << std::endl;
#endif
}

size_t point_decoration(const std::vector<Pillar> &bev_pillar,
                        std::vector<Voxel> &voxels, const float *points,
                        size_t points_buf_len, size_t point_stride) {
    size_t num_pillars = 0;
    for (Pillar pillar : bev_pillar) {
        if (pillar.is_empty) {
            continue;
        }
        // calc mean values for all points in current pillar
        float mean_x = 0.0f;
        float mean_y = 0.0f;
        float mean_z = 0.0f;

        // double check grid index of current pillar
        assert(pillar.pillar_grid_x < GRID_X_SIZE &&
               pillar.pillar_grid_y < GRID_Y_SIZE);

        for (size_t i = 0; i < pillar.point_num_in_pillar; i++) {
            size_t point_index = pillar.point_index[i];
            float curr_x = points[point_stride * point_index];
            float curr_y = points[point_stride * point_index + 1];
            float curr_z = points[point_stride * point_index + 2];
            mean_x += curr_x;
            mean_y += curr_y;
            mean_z += curr_z;

            // double check if current point is out-of-range point
            assert(curr_x >= MIN_X_RANGE && curr_x <= MAX_X_RANGE &&
                   curr_y >= MIN_Y_RANGE && curr_y <= MAX_Y_RANGE &&
                   curr_z >= MIN_Z_RANGE && curr_z <= MAX_Z_RANGE);
        }
        mean_x /= pillar.point_num_in_pillar;
        mean_y /= pillar.point_num_in_pillar;
        mean_z /= pillar.point_num_in_pillar;

        // calc centeral values of current pillar
        float x_center = (VOXEL_X_SIZE / 2.0f) +
                         (pillar.pillar_grid_x * VOXEL_X_SIZE) + MIN_X_RANGE;
        float y_center = (VOXEL_Y_SIZE / 2.0f) +
                         (pillar.pillar_grid_y * VOXEL_Y_SIZE) + MIN_Y_RANGE;
        float z_center = (VOXEL_Z_SIZE / 2.0f) + MIN_Z_RANGE;

        // write encoded features into Voxel
        for (size_t i = 0; i < pillar.point_num_in_pillar; i++) {
            size_t point_index = pillar.point_index[i];
            size_t voxel_index = i + MAX_NUM_POINTS_PER_PILLAR *
                                         (pillar.pillar_grid_x +
                                          pillar.pillar_grid_y * GRID_X_SIZE);
            voxels[voxel_index].x = points[point_stride * point_index];
            voxels[voxel_index].y = points[point_stride * point_index + 1];
            voxels[voxel_index].z = points[point_stride * point_index + 2];
#if NUM_POINT_VALUES >= 4
#ifdef ZERO_INTENSITY
            voxels[voxel_index].w = 0.0f;
#else
            voxels[voxel_index].w = points[point_stride * point_index + 3] /
                                    INTENSITY_NORMALIZE_DIV;
#endif
#endif
            voxels[voxel_index].offset_from_mean_x =
                voxels[voxel_index].x - mean_x;
            voxels[voxel_index].offset_from_mean_y =
                voxels[voxel_index].y - mean_y;
            voxels[voxel_index].offset_from_mean_z =
                voxels[voxel_index].z - mean_z;

            voxels[voxel_index].offset_from_center_x =
                voxels[voxel_index].x - x_center;
            voxels[voxel_index].offset_from_center_y =
                voxels[voxel_index].y - y_center;
            voxels[voxel_index].offset_from_center_z =
                voxels[voxel_index].z - z_center;
            voxels[voxel_index].grid_x = pillar.pillar_grid_x;
            voxels[voxel_index].grid_y = pillar.pillar_grid_y;
            voxels[voxel_index].is_valid = true;
        }
        num_pillars++;
    }

    return num_pillars;
}

size_t gather(const std::vector<Voxel> &raw_voxels,
              std::vector<float> &pfe_input) {
    size_t index = 0;
    assert(pfe_input.size() ==
           MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM);
    for (Voxel voxel : raw_voxels) {
        if (voxel.is_valid) {
            pfe_input[index] = voxel.x;
            pfe_input[index + 1] = voxel.y;
            pfe_input[index + 2] = voxel.z;
#if NUM_POINT_VALUES >= 4
            pfe_input[index + 3] = voxel.w;
            pfe_input[index + 4] = voxel.offset_from_mean_x;
            pfe_input[index + 5] = voxel.offset_from_mean_y;
            pfe_input[index + 6] = voxel.offset_from_mean_z;
            pfe_input[index + 7] = voxel.offset_from_center_x;
            pfe_input[index + 8] = voxel.offset_from_center_y;
            pfe_input[index + 9] = voxel.offset_from_center_z;
            assert(FEATURE_NUM == 10);
#else
            pfe_input[index + 3] = voxel.offset_from_mean_x;
            pfe_input[index + 4] = voxel.offset_from_mean_y;
            pfe_input[index + 5] = voxel.offset_from_mean_z;
            pfe_input[index + 6] = voxel.offset_from_center_x;
            pfe_input[index + 7] = voxel.offset_from_center_y;
            pfe_input[index + 8] = voxel.offset_from_center_z;
            assert(FEATURE_NUM == 9)
#endif
            index += FEATURE_NUM;
        }
    }
#ifdef _DEBUG
    std::cout << index << std::endl;
#endif

    return index / FEATURE_NUM;
}

// TODO: Implement here
void scatter(const std::vector<Voxel> &raw_voxels,
             std::vector<float> &bev_image, size_t num_valid_voxels) {
    assert(std::accumulate(bev_image.begin(), bev_image.end(), 0.0f) == 0.0f);
    for (Voxel voxel : raw_voxels) {
        size_t grid_x = voxel.grid_x;
        size_t grid_y = voxel.grid_y;
        size_t in_bev_index = grid_y * GRID_X_SIZE + grid_x;
    }
}

void run(const std::vector<float> &pfe_input, std::vector<float> &pfe_output) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, PFE_PATH, session_options);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // const size_t num_input_nodes = session.GetInputCount();
    std::vector<int64_t> input_node_dims = {
        MAX_VOXELS, MAX_NUM_POINTS_PER_PILLAR, FEATURE_NUM};
    size_t input_tensor_size =
        MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM;

    // create input tensor object from data values
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float *)pfe_input.data(), input_tensor_size,
        input_node_dims.data(), 3);
    assert(input_tensor.IsTensor());

    std::vector<const char *> input_node_names = {"voxels"};
    std::vector<const char *> output_node_names = {"pfe_output"};

    // score model & input tensor, get back output tensor
    auto output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                    &input_tensor, 1, output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    // Get pointer to output tensor float values
    // Get the first (and assumed to be only) output tensor
    Ort::Value &output_tensor = output_tensors.front();

    // Get the shape of the output tensor
    auto output_tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    size_t output_size = output_tensor_info.GetElementCount();
    float *floatarr = output_tensor.GetTensorMutableData<float>();

    // Resize the output vector to fit the output tensor data
    pfe_output.resize(output_size);

    // Copy the output tensor data to the output vector
    std::copy(floatarr, floatarr + output_size, pfe_output.begin());
    std::cout << "INFERENCE DONE." << std::endl;
}

void preprocess(const float *points, size_t points_buf_len,
                size_t point_stride) {
    std::vector<Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
    std::vector<Voxel> raw_voxels(GRID_Y_SIZE * GRID_X_SIZE *
                                  MAX_NUM_POINTS_PER_PILLAR);
    std::vector<float> pfe_input(
        MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f);
    std::vector<float> bev_image(
        GRID_Y_SIZE * GRID_X_SIZE * RPN_INPUT_NUM_CHANNELS, 0.0f);
    std::vector<float> pfe_output;
    voxelization(bev_pillar, points, points_buf_len, point_stride);
    size_t num_pillars = point_decoration(bev_pillar, raw_voxels, points,
                                          points_buf_len, point_stride);
    size_t num_valid_voxels = gather(raw_voxels, pfe_input);
    run(pfe_input, pfe_output);
    assert(pfe_output.size() == MAX_VOXELS * RPN_INPUT_NUM_CHANNELS);
}

} // namespace vueron
#endif // PRE_H
