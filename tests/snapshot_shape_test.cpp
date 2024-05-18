#include "npy.h"
#include "params.h"
#include "pcl.h"
#include "pre.h"
#include "shape.h"
#include "utils.h"
#include <glob.h>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

TEST(VoxelSnapshotTest, VoxelShapeTest) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    std::vector<float> points;

    for (const auto &file : pcd_files) {
        points = vueron::readPcdFile(file, MAX_POINTS_NUM);
        vueron::preprocess((float *)points.data(), points.size(),
                           sizeof(float));
    }

    // read snapshot
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files =
        vueron::getFileList(snapshot_folder_path);

    for (std::string snapshot_dir : snapshot_files) {
        const std::string voxels_path = snapshot_dir + "/voxels.npy";
        const std::string voxels_encoded_path =
            snapshot_dir + "/voxels_encoded.npy";
        const std::string voxel_coord_path = snapshot_dir + "/voxel_coord.npy";
        const std::string voxel_num_points_path =
            snapshot_dir + "/voxel_num_points.npy";
        auto raw_voxels = npy::read_npy<float>(voxels_path);
        auto raw_voxels_encoded = npy::read_npy<float>(voxels_encoded_path);
        auto raw_voxel_coord = npy::read_npy<uint32_t>(voxel_coord_path);
        auto raw_voxel_num_points =
            npy::read_npy<uint32_t>(voxel_num_points_path);

        std::vector<float> voxels = raw_voxels.data;
        std::vector<float> voxels_encoded = raw_voxels_encoded.data;
        std::vector<uint32_t> voxel_coord = raw_voxel_coord.data;
        std::vector<uint32_t> voxel_num_points = raw_voxel_num_points.data;

        std::vector<unsigned long> voxel_shape = raw_voxels.shape;
        std::vector<unsigned long> voxels_encoded_shape =
            raw_voxels_encoded.shape;
        std::vector<unsigned long> voxel_coord_shape = raw_voxel_coord.shape;
        std::vector<unsigned long> voxel_num_points_shape =
            raw_voxel_num_points.shape;

        // check number of pillars
        EXPECT_EQ(voxel_num_points_shape[0], voxel_shape[0]);
        EXPECT_EQ(voxel_num_points_shape[0], voxel_coord_shape[0]);
        EXPECT_EQ(voxel_shape[0], voxel_coord_shape[0]);
        EXPECT_EQ(voxels_encoded_shape[0], voxel_coord_shape[0]);

        // check Tensor shape
        EXPECT_EQ(voxels.size(), voxel_shape[0] * MAX_NUM_POINTS_PER_PILLAR *
                                     NUM_POINT_VALUES);
        EXPECT_EQ(voxel_coord.size(), voxel_coord_shape[0] * 2);
        EXPECT_EQ(voxels_encoded.size(), voxels_encoded_shape[0] *
                                             MAX_NUM_POINTS_PER_PILLAR *
                                             FEATURE_NUM);
        EXPECT_EQ(voxels.size() / MAX_NUM_POINTS_PER_PILLAR / NUM_POINT_VALUES,
                  voxel_coord.size() / 2);
        EXPECT_EQ(voxels_encoded.size() / MAX_NUM_POINTS_PER_PILLAR /
                      FEATURE_NUM,
                  voxel_coord.size() / 2);
        EXPECT_EQ(voxel_num_points.size(), voxel_coord.size() / 2);
    }
}

TEST(VoxelSnapshotTest, VoxelValueTest) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files =
        vueron::getFileList(snapshot_folder_path);
    std::vector<float> points;
    size_t num_test_files = pcd_files.size();

    assert(pcd_files.size() == snapshot_files.size());

    for (size_t i = 0; i < num_test_files; i++) {
        std::string pcd_file = pcd_files[i];
        std::string snapshot_dir = snapshot_files[i];
        std::cout << "Testing : " << pcd_file << std::endl;

        // read point from pcd file
        points = vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);

        // read voxels from snapshot file
        const std::string voxels_path = snapshot_dir + "/voxels.npy";
        const std::string voxels_encoded_path =
            snapshot_dir + "/voxels_encoded.npy";
        const std::string voxel_coord_path = snapshot_dir + "/voxel_coord.npy";
        const std::string voxel_num_points_path =
            snapshot_dir + "/voxel_num_points.npy";
        auto raw_voxels = npy::read_npy<float>(voxels_path);
        auto raw_voxels_encoded = npy::read_npy<float>(voxels_encoded_path);
        auto raw_voxel_coord = npy::read_npy<uint32_t>(voxel_coord_path);
        auto raw_voxel_num_points =
            npy::read_npy<uint32_t>(voxel_num_points_path);

        std::vector<float> voxels = raw_voxels.data;
        std::vector<float> voxels_encoded = raw_voxels_encoded.data;
        std::vector<uint32_t> voxel_coord = raw_voxel_coord.data;
        std::vector<uint32_t> voxel_num_points = raw_voxel_num_points.data;

        std::vector<unsigned long> voxels_shape = raw_voxels.shape;
        std::vector<unsigned long> voxels_encoded_shape =
            raw_voxels_encoded.shape;
        std::vector<unsigned long> voxel_coord_shape = raw_voxel_coord.shape;
        std::vector<unsigned long> voxel_num_points_shape =
            raw_voxel_num_points.shape;

        std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
        std::vector<vueron::Voxel> bev_voxels(GRID_Y_SIZE * GRID_X_SIZE *
                                              MAX_NUM_POINTS_PER_PILLAR);
        std::vector<float> pfe_input(
            MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f);

        vueron::voxelization(bev_pillar, (float *)points.data(), points.size(),
                             sizeof(float));
        size_t num_pillars = vueron::point_decoration(
            bev_pillar, bev_voxels, (float *)points.data(), points.size(),
            sizeof(float));

        EXPECT_EQ(num_pillars,
                  voxels.size() / MAX_NUM_POINTS_PER_PILLAR / NUM_POINT_VALUES);
        size_t num_valid_voxels = vueron::gather(bev_voxels, pfe_input);
        size_t num_valid_voxels_snapshot = std::accumulate(
            voxel_num_points.begin(), voxel_num_points.end(), 0);
        EXPECT_EQ(num_valid_voxels, num_valid_voxels_snapshot);

        // check each pillars
        for (size_t j = 0; j < voxel_num_points.size(); j++) {
            size_t grid_x = (size_t)voxel_coord[2 * j + 1]; // cols
            size_t grid_y = (size_t)voxel_coord[2 * j];     // rows
            size_t voxel_index = grid_y * GRID_X_SIZE + grid_x;

            // check num_points of current pillar
            vueron::Pillar curr_pillar = bev_pillar[voxel_index];
            EXPECT_EQ(curr_pillar.point_num_in_pillar, voxel_num_points[j]);
            EXPECT_FALSE(curr_pillar.is_empty);
            EXPECT_LT(voxel_index, GRID_Y_SIZE * GRID_X_SIZE);
            EXPECT_LT(grid_x, GRID_X_SIZE);
            EXPECT_LT(grid_y, GRID_Y_SIZE);
            EXPECT_EQ(grid_x, curr_pillar.pillar_grid_x);
            EXPECT_EQ(grid_y, curr_pillar.pillar_grid_y);

            // check each point values in current pillar
            for (size_t k = 0; k < MAX_NUM_POINTS_PER_PILLAR; k++) {
                size_t acture_points_num = voxel_num_points[j];
                size_t in_voxel_index =
                    voxel_index * MAX_NUM_POINTS_PER_PILLAR + k;
                vueron::Voxel curr_voxel;
                if (k < acture_points_num) {
                    curr_voxel = bev_voxels[in_voxel_index];
                    EXPECT_TRUE(curr_voxel.is_valid);
                    EXPECT_EQ(curr_voxel.grid_x, grid_x);
                    EXPECT_EQ(curr_voxel.grid_y, grid_y);
                }
                EXPECT_LT(in_voxel_index, GRID_Y_SIZE * GRID_X_SIZE *
                                              MAX_NUM_POINTS_PER_PILLAR);
                size_t offset = j * FEATURE_NUM * MAX_NUM_POINTS_PER_PILLAR +
                                FEATURE_NUM * k;
                std::vector<float> vec_curr_voxel_encoded_ref(
                    offset + voxels_encoded.begin(),
                    offset + voxels_encoded.begin() + FEATURE_NUM);
                float *curr_voxel_encoded_ref =
                    (float *)vec_curr_voxel_encoded_ref.data();
                EXPECT_NEAR(curr_voxel_encoded_ref[0], curr_voxel.x, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[1], curr_voxel.y, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[2], curr_voxel.z, _EPSILON);
#if NUM_POINT_VALUES >= 4
                EXPECT_NEAR(curr_voxel_encoded_ref[3] / INTENSITY_NORMALIZE_DIV,
                            curr_voxel.w, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[4],
                            curr_voxel.offset_from_mean_x, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[5],
                            curr_voxel.offset_from_mean_y, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[6],
                            curr_voxel.offset_from_mean_z, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[7],
                            curr_voxel.offset_from_center_x, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[8],
                            curr_voxel.offset_from_center_y, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[9],
                            curr_voxel.offset_from_center_z, _EPSILON);
#else
                EXPECT_NEAR(curr_voxel_encoded_ref[3],
                            curr_voxel.offset_from_mean_x, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[4],
                            curr_voxel.offset_from_mean_y, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[5],
                            curr_voxel.offset_from_mean_z, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[6],
                            curr_voxel.offset_from_center_x, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[7],
                            curr_voxel.offset_from_center_y, _EPSILON);
                EXPECT_NEAR(curr_voxel_encoded_ref[8],
                            curr_voxel.offset_from_center_z, _EPSILON);
#endif
            }
        }
        std::cout << "Test Finish : " << pcd_file << std::endl;
    }
}

TEST(VoxelSnapshotTest, VoxelGatherTest) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files =
        vueron::getFileList(snapshot_folder_path);
    std::vector<float> points;
    size_t num_test_files = pcd_files.size();

    assert(pcd_files.size() == snapshot_files.size());

    for (size_t i = 0; i < num_test_files; i++) {
        std::string pcd_file = pcd_files[i];
        std::string snapshot_dir = snapshot_files[i];
        std::cout << "Testing : " << pcd_file << std::endl;

        // read point from pcd file
        points = vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);

        // read voxels from snapshot file
        const std::string voxels_path = snapshot_dir + "/voxels.npy";
        const std::string voxels_encoded_path =
            snapshot_dir + "/voxels_encoded.npy";
        const std::string voxel_coord_path = snapshot_dir + "/voxel_coord.npy";
        const std::string voxel_num_points_path =
            snapshot_dir + "/voxel_num_points.npy";
        auto raw_voxels = npy::read_npy<float>(voxels_path);
        auto raw_voxels_encoded = npy::read_npy<float>(voxels_encoded_path);
        auto raw_voxel_coord = npy::read_npy<uint32_t>(voxel_coord_path);
        auto raw_voxel_num_points =
            npy::read_npy<uint32_t>(voxel_num_points_path);

        std::vector<float> voxels = raw_voxels.data;
        std::vector<float> voxels_encoded = raw_voxels_encoded.data;
        std::vector<uint32_t> voxel_coord = raw_voxel_coord.data;
        std::vector<uint32_t> voxel_num_points = raw_voxel_num_points.data;

        std::vector<unsigned long> voxels_shape = raw_voxels.shape;
        std::vector<unsigned long> voxels_encoded_shape =
            raw_voxels_encoded.shape;
        std::vector<unsigned long> voxel_coord_shape = raw_voxel_coord.shape;
        std::vector<unsigned long> voxel_num_points_shape =
            raw_voxel_num_points.shape;

        std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
        std::vector<vueron::Voxel> bev_voxels(GRID_Y_SIZE * GRID_X_SIZE *
                                              MAX_NUM_POINTS_PER_PILLAR);
        std::vector<float> pfe_input(
            MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f);

        vueron::voxelization(bev_pillar, (float *)points.data(), points.size(),
                             sizeof(float));
        size_t num_pillars = vueron::point_decoration(
            bev_pillar, bev_voxels, (float *)points.data(), points.size(),
            sizeof(float));

        EXPECT_EQ(num_pillars,
                  voxels.size() / MAX_NUM_POINTS_PER_PILLAR / NUM_POINT_VALUES);
        size_t num_valid_voxels = vueron::gather(bev_voxels, pfe_input);
        size_t num_valid_voxels_snapshot = std::accumulate(
            voxel_num_points.begin(), voxel_num_points.end(), 0);
        EXPECT_EQ(num_valid_voxels, num_valid_voxels_snapshot);
        float sum_of_padded_voxels =
            std::accumulate(pfe_input.begin() + num_valid_voxels * FEATURE_NUM,
                            pfe_input.end(), 0.0f);
        EXPECT_FLOAT_EQ(sum_of_padded_voxels, 0.0f);
        std::cout << "Test Finish : " << pcd_file << std::endl;
    }
}

TEST(VoxelSnapshotTest, GatheredVoxelValueTest) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files =
        vueron::getFileList(snapshot_folder_path);
    std::vector<float> points;
    size_t num_test_files = pcd_files.size();

    assert(pcd_files.size() == snapshot_files.size());

    for (size_t i = 0; i < num_test_files; i++) {
        std::string pcd_file = pcd_files[i];
        std::string snapshot_dir = snapshot_files[i];
        std::cout << "Testing : " << pcd_file << std::endl;

        // read point from pcd file
        points = vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);

        // read voxels from snapshot file
        const std::string voxels_path = snapshot_dir + "/voxels.npy";
        const std::string voxels_encoded_path =
            snapshot_dir + "/voxels_encoded.npy";
        const std::string voxel_coord_path = snapshot_dir + "/voxel_coord.npy";
        const std::string voxel_num_points_path =
            snapshot_dir + "/voxel_num_points.npy";
        auto raw_voxels = npy::read_npy<float>(voxels_path);
        auto raw_voxels_encoded = npy::read_npy<float>(voxels_encoded_path);
        auto raw_voxel_coord = npy::read_npy<uint32_t>(voxel_coord_path);
        auto raw_voxel_num_points =
            npy::read_npy<uint32_t>(voxel_num_points_path);

        std::vector<float> voxels = raw_voxels.data;
        std::vector<float> voxels_encoded = raw_voxels_encoded.data;
        std::vector<uint32_t> voxel_coord = raw_voxel_coord.data;
        std::vector<uint32_t> voxel_num_points = raw_voxel_num_points.data;

        std::vector<unsigned long> voxels_shape = raw_voxels.shape;
        std::vector<unsigned long> voxels_encoded_shape =
            raw_voxels_encoded.shape;
        std::vector<unsigned long> voxel_coord_shape = raw_voxel_coord.shape;
        std::vector<unsigned long> voxel_num_points_shape =
            raw_voxel_num_points.shape;

        std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
        std::vector<vueron::Voxel> bev_voxels(GRID_Y_SIZE * GRID_X_SIZE *
                                              MAX_NUM_POINTS_PER_PILLAR);
        std::vector<float> pfe_input(
            MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f);

        vueron::voxelization(bev_pillar, (float *)points.data(), points.size(),
                             sizeof(float));
        size_t num_pillars = vueron::point_decoration(
            bev_pillar, bev_voxels, (float *)points.data(), points.size(),
            sizeof(float));

        EXPECT_EQ(num_pillars,
                  voxels.size() / MAX_NUM_POINTS_PER_PILLAR / NUM_POINT_VALUES);
        size_t num_valid_voxels = vueron::gather(bev_voxels, pfe_input);

        // sort gathered encoded voxels
        std::vector<float> pfe_x_values;
        std::vector<float> pfe_x_values_sorted;
        std::vector<float> pfe_x_values_snapshot;
        std::vector<float> pfe_x_values_snapshot_sorted;
        std::vector<float> gathered_voxels_sorted;
        std::vector<float> gathered_voxels_sorted_snapshot;

        // copy x values and sort it
        for (size_t idx_x = 0; idx_x < pfe_input.size(); idx_x += FEATURE_NUM) {
            pfe_x_values.push_back(pfe_input[idx_x]);
            pfe_x_values_sorted.push_back(pfe_input[idx_x]);
            EXPECT_LT(idx_x,
                      MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM);
        }
        std::sort(pfe_x_values_sorted.begin(), pfe_x_values_sorted.end(),
                  std::greater<>());

        for (size_t idx_x = 0; idx_x < voxels_encoded.size();
             idx_x += FEATURE_NUM) {
            pfe_x_values_snapshot.push_back(voxels_encoded[idx_x]);
            pfe_x_values_snapshot_sorted.push_back(voxels_encoded[idx_x]);
            EXPECT_LT(idx_x,
                      MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM);
        }
        std::sort(pfe_x_values_snapshot_sorted.begin(),
                  pfe_x_values_snapshot_sorted.end(), std::greater<>());
        EXPECT_FLOAT_EQ(std::accumulate(pfe_x_values_snapshot_sorted.begin() +
                                            pfe_x_values_snapshot_sorted.size(),
                                        pfe_x_values_snapshot_sorted.end(),
                                        0.0f),
                        0.0f);
        EXPECT_TRUE(pfe_x_values.size() == pfe_x_values_sorted.size());
        EXPECT_TRUE(pfe_x_values_snapshot.size() ==
                    pfe_x_values_snapshot_sorted.size());
        // EXPECT_EQ(pfe_x_values_snapshot.size(), pfe_x_values.size());

        // check two x values are equal or not
        for (size_t idx_x = 0; idx_x < pfe_x_values_snapshot.size(); idx_x++) {
            EXPECT_FLOAT_EQ(pfe_x_values_snapshot_sorted[idx_x],
                            pfe_x_values_sorted[idx_x]);
        }
#if 0
        for (float x_value : pfe_x_values_snapshot_sorted) {
            int voxel_index = vueron::find_index(pfe_x_values, x_value);
            int voxel_index_snapshot =
                vueron::find_index(pfe_x_values_snapshot, x_value);
            int target_index =
                voxel_index * MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR;
            int target_index_snapshot =
                voxel_index * MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR;
            EXPECT_TRUE(voxel_index > 0);
            EXPECT_TRUE(voxel_index_snapshot > 0);

            for (size_t j = 0; j < FEATURE_NUM; j++) {
                gathered_voxels_sorted.push_back(pfe_input[target_index]);
                gathered_voxels_sorted_snapshot.push_back(
                    voxels_encoded[target_index_snapshot]);
            }
        }

        EXPECT_EQ(gathered_voxels_sorted.size(),
                  gathered_voxels_sorted_snapshot.size());

        for (size_t j = 0; j < gathered_voxels_sorted.size(); j++) {
            EXPECT_NEAR(gathered_voxels_sorted[j],
                        gathered_voxels_sorted_snapshot[j], _EPSILON);
        }
#endif

#if 0
        // sort gathered encoded voxels
        std::vector<float> pfe_x_values;
        std::vector<float> gathered_voxels_sorted(
            MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f);
        for (size_t idx_x = 0; idx_x < pfe_input.size(); idx_x += FEATURE_NUM) {
            pfe_x_values.push_back(pfe_input[idx_x]);
            EXPECT_LT(idx_x,
                      MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM);
        }
        EXPECT_EQ(pfe_x_values.size() * FEATURE_NUM, pfe_input.size());
        std::vector<size_t> voxels_sorted_index =
            vueron::sort_and_get_indices(pfe_x_values);
        EXPECT_EQ(voxels_sorted_index.size(), pfe_x_values.size());
        for (size_t idx_x = 0; idx_x < voxels_sorted_index.size(); idx_x++) {
            for (size_t feature_idx = 0; feature_idx < FEATURE_NUM;
                 feature_idx++) {
                size_t target_idx = idx_x * FEATURE_NUM + feature_idx;
                size_t source_idx =
                    voxels_sorted_index[idx_x] * FEATURE_NUM + feature_idx;
                gathered_voxels_sorted[target_idx] = pfe_input[source_idx];
            }
        }
        // sort snapshot encoded voxels
        std::vector<float> pfe_x_values_snapshot;
        std::vector<float> gathered_voxels_sorted_snapshot(
            MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f);
        for (size_t idx_x = 0; idx_x < voxels_encoded.size();
             idx_x += FEATURE_NUM) {
            pfe_x_values_snapshot.push_back(voxels_encoded[idx_x]);
            EXPECT_LT(idx_x,
                      MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM);
        }
        EXPECT_EQ(pfe_x_values_snapshot.size() * FEATURE_NUM,
                  voxels_encoded.size());
        std::vector<size_t> voxels_sorted_index_snapshot =
            vueron::sort_and_get_indices(pfe_x_values_snapshot);
        EXPECT_EQ(voxels_sorted_index_snapshot.size(),
                  pfe_x_values_snapshot.size());
        for (size_t idx_x = 0; idx_x < voxels_sorted_index_snapshot.size();
             idx_x++) {
            for (size_t feature_idx = 0; feature_idx < FEATURE_NUM;
                 feature_idx++) {
                size_t target_idx = idx_x * FEATURE_NUM + feature_idx;
                size_t source_idx =
                    voxels_sorted_index_snapshot[idx_x] * FEATURE_NUM +
                    feature_idx;
                gathered_voxels_sorted_snapshot[target_idx] =
                    voxels_encoded[source_idx];
            }
        }
        EXPECT_EQ(gathered_voxels_sorted_snapshot.size(),
                  gathered_voxels_sorted.size());
        for (size_t idx_x = 0; idx_x < voxels_encoded.size(); idx_x++) {
            EXPECT_FLOAT_EQ(gathered_voxels_sorted_snapshot[idx_x],
                            gathered_voxels_sorted[idx_x]);
        }
#endif

        std::cout << "Test Finish : " << pcd_file << std::endl;
    }
}
