#include "npy.h"
#include "params.h"
#include "pcl.h"
#include "pre.h"
#include "shape.h"
#include <glob.h>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

TEST(VoxelSnapshotTest, VoxelShapeTest) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = getFileList(folder_path);
    std::vector<float> points;

    for (const auto &file : pcd_files) {
        points = readPcdFile(file, MAX_POINTS_NUM);
        vueron::preprocess((float *)points.data(), points.size(),
                           sizeof(float));
    }

    // read snapshot
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files = getFileList(snapshot_folder_path);

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
    std::vector<std::string> pcd_files = getFileList(folder_path);
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files = getFileList(snapshot_folder_path);
    std::vector<float> points;
    size_t num_test_files = pcd_files.size();

    assert(pcd_files.size() == snapshot_files.size());

    for (size_t i = 0; i < num_test_files; i++) {
        std::string pcd_file = pcd_files[i];
        std::string snapshot_dir = snapshot_files[i];
        std::cout << "Testing : " << pcd_file << std::endl;

        // read point from pcd file
        points = readPcdFile(pcd_file, MAX_POINTS_NUM);

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
    std::vector<std::string> pcd_files = getFileList(folder_path);
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files = getFileList(snapshot_folder_path);
    std::vector<float> points;
    size_t num_test_files = pcd_files.size();

    assert(pcd_files.size() == snapshot_files.size());

    for (size_t i = 0; i < num_test_files; i++) {
        std::string pcd_file = pcd_files[i];
        std::string snapshot_dir = snapshot_files[i];
        std::cout << "Testing : " << pcd_file << std::endl;

        // read point from pcd file
        points = readPcdFile(pcd_file, MAX_POINTS_NUM);

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
