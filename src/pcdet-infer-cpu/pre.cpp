#include "pcdet-infer-cpu/pre.h"

void vueron::voxelization(std::vector<Pillar> &bev_pillar, const float *points,
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
    std::iota(indices.begin(), indices.end(), 0);

#if _SHIFFLE == ON
    std::shuffle(indices.begin(), indices.end(), rng);
#endif
    size_t num_points_to_voxelize =
        (points_num > MAX_POINTS_NUM) ? MAX_POINTS_NUM : points_num;
    for (size_t idx = 0; idx < num_points_to_voxelize; idx++) {
        size_t i = indices[idx];
        float point_x = points[point_stride * i];
        float point_y = points[point_stride * i + 1];
        float point_z = points[point_stride * i + 2];

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
        if (bev_pillar[voxel_index].point_num_in_pillar <
            MAX_NUM_POINTS_PER_PILLAR) {
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
}

size_t vueron::point_decoration(const std::vector<Pillar> &bev_pillar,
                                std::vector<size_t> &voxel_coords,
                                std::vector<size_t> &voxel_num_points,
                                std::vector<float> &pfe_input,
                                const float *points, size_t points_buf_len,
                                size_t point_stride) {
    size_t num_pillars = 0;
    size_t index = 0;

    for (Pillar pillar : bev_pillar) {
        if (pillar.is_empty) {
            continue;
        }
        assert(pillar.pillar_grid_x < GRID_X_SIZE);
        assert(pillar.pillar_grid_y < GRID_Y_SIZE);

        // calc mean values for all points in current pillar
        float mean_x = 0.0f;
        float mean_y = 0.0f;
        float mean_z = 0.0f;

        // double check grid index of current pillar
        assert(pillar.pillar_grid_x < GRID_X_SIZE &&
               pillar.pillar_grid_y < GRID_Y_SIZE);

        voxel_coords.push_back(pillar.pillar_grid_x);
        voxel_coords.push_back(pillar.pillar_grid_y);
        voxel_num_points.push_back(pillar.point_num_in_pillar);

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

        // write encoded features into raw_voxels and pfe_input
        for (size_t i = 0; i < MAX_NUM_POINTS_PER_PILLAR; i++) {
            if (i < pillar.point_num_in_pillar) {
                size_t point_index = pillar.point_index[i];
                size_t voxel_index =
                    i + MAX_NUM_POINTS_PER_PILLAR *
                            (pillar.pillar_grid_x +
                             pillar.pillar_grid_y * GRID_X_SIZE);
                pfe_input[index] = points[point_stride * point_index];
                pfe_input[index + 1] = points[point_stride * point_index + 1];
                pfe_input[index + 2] = points[point_stride * point_index + 2];
#if NUM_POINT_VALUES >= 4
#ifdef ZERO_INTENSITY
                pfe_input[index + 3] = 0.0f;
#else
                pfe_input[index + 3] = points[point_stride * point_index + 3] /
                                       INTENSITY_NORMALIZE_DIV;
#endif
                pfe_input[index + 4] = pfe_input[index] - mean_x;
                pfe_input[index + 5] = pfe_input[index + 1] - mean_y;
                pfe_input[index + 6] = pfe_input[index + 2] - mean_z;

                pfe_input[index + 7] = pfe_input[index] - x_center;
                pfe_input[index + 8] = pfe_input[index + 1] - y_center;
                pfe_input[index + 9] = pfe_input[index + 2] - z_center;
#else
                // requires valiation here
                pfe_input[index + 3] = pfe_input[index] - mean_x;
                pfe_input[index + 4] = pfe_input[index + 1] - mean_y;
                pfe_input[index + 5] = pfe_input[index + 2] - mean_z;

                pfe_input[index + 6] = pfe_input[index] - x_center;
                pfe_input[index + 7] = pfe_input[index + 1] - y_center;
                pfe_input[index + 8] = pfe_input[index + 2] - z_center;
#endif
            }
            index += FEATURE_NUM;
        }
        num_pillars++;
    }

    assert(voxel_coords.size() / 2 == voxel_num_points.size());

    return num_pillars;
}

void vueron::pfe_run(const std::vector<float> &pfe_input,
                     std::vector<float> &pfe_output) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, PFE_PATH, session_options);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

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
    auto output_dims_count = output_tensor_info.GetDimensionsCount();
    size_t output_size = output_tensor_info.GetElementCount();
    float *floatarr = output_tensor.GetTensorMutableData<float>();

    assert(output_size == MAX_VOXELS * NUM_FEATURE_SCATTER);
    assert(output_dims.size() == output_dims_count && output_dims_count == 2);
    assert(output_dims[0] == MAX_VOXELS);
    assert(output_dims[1] == NUM_FEATURE_SCATTER);

    // Resize the output vector to fit the output tensor data
    pfe_output.resize(output_size);

    // Copy the output tensor data to the output vector
    std::copy(floatarr, floatarr + output_size, pfe_output.begin());
#ifdef _DEBUG
    std::cout << " PFE INFERENCE DONE." << std::endl;
#endif
}

void vueron::scatter(const std::vector<float> &pfe_output,
                     const std::vector<size_t> &voxel_coords,
                     size_t num_pillars, std::vector<float> &rpn_input) {
    assert(rpn_input.size() == GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER);
    assert(pfe_output.size() == MAX_VOXELS * NUM_FEATURE_SCATTER);

    for (size_t i = 0; i < num_pillars; i++) {
        // voxel_coords : (x, y)
        size_t curr_grid_x = voxel_coords[2 * i];
        size_t curr_grid_y = voxel_coords[2 * i + 1];
        size_t source_voxel_index = NUM_FEATURE_SCATTER * i;
        assert(source_voxel_index < MAX_VOXELS * NUM_FEATURE_SCATTER);

        for (size_t j = 0; j < NUM_FEATURE_SCATTER; j++) {
            size_t target_voxel_index = (j * GRID_Y_SIZE * GRID_X_SIZE) +
                                        (curr_grid_y * GRID_X_SIZE) +
                                        curr_grid_x;
            assert(target_voxel_index <
                   GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER);
            rpn_input[target_voxel_index] = pfe_output[source_voxel_index + j];
        }
    }
}