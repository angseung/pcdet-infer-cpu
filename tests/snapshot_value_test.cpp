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

TEST(VoxelSnapshotTest, BEVValueTest) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files =
        vueron::getFileList(snapshot_folder_path);
    std::vector<float> points;
    size_t num_test_files = pcd_files.size();

    EXPECT_GE(pcd_files.size(), snapshot_files.size());

    for (size_t i = 0; i < num_test_files; i++) {
        std::string pcd_file = pcd_files[i];
        std::string snapshot_dir = snapshot_files[i];
        std::cout << "Testing : " << pcd_file << std::endl;

        // read point from pcd file
        std::vector<float> points =
            vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);
        size_t points_buf_len = points.size();
        size_t point_stride = sizeof(float);
        std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
        std::vector<size_t> voxel_coords; // (x, y)
        std::vector<size_t> voxel_num_points;
        std::vector<vueron::Voxel> raw_voxels(
            GRID_Y_SIZE * GRID_X_SIZE *
            MAX_NUM_POINTS_PER_PILLAR); // input of gather()
        std::vector<float> pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR *
                                         FEATURE_NUM,
                                     0.0f); // input of run()
        std::vector<float> pfe_output(MAX_VOXELS * RPN_INPUT_NUM_CHANNELS,
                                      0.0f); // input of scatter()
        std::vector<float> bev_image(GRID_Y_SIZE * GRID_X_SIZE *
                                         RPN_INPUT_NUM_CHANNELS,
                                     0.0f); // input of RPN
        vueron::voxelization(bev_pillar, (float *)points.data(), points_buf_len,
                             point_stride);
        size_t num_pillars = vueron::point_decoration(
            bev_pillar, voxel_coords, voxel_num_points, raw_voxels,
            (float *)points.data(), points_buf_len, point_stride);
        size_t num_valid_voxels = vueron::gather(raw_voxels, pfe_input);
        size_t num_voxels_manual = std::accumulate(voxel_num_points.begin(),
                                                   voxel_num_points.end(), 0);
        EXPECT_EQ(num_voxels_manual, num_valid_voxels);
        EXPECT_EQ(num_pillars, voxel_num_points.size());

        // for (size_t j = 0; j < voxel_num_points.size(); j++) {
        //     size_t curr_num_points = voxel_num_points[j];
        // }
        vueron::run(pfe_input, pfe_output);
        EXPECT_EQ(pfe_output.size(), MAX_VOXELS * RPN_INPUT_NUM_CHANNELS);

        // check remainder voxels is zero
        size_t padded_voxel_offset = num_valid_voxels * FEATURE_NUM;
        float remainder_sum = std::accumulate(
            pfe_input.begin() + padded_voxel_offset, pfe_input.end(), 0.0f);
        assert(remainder_sum == 0.0f);

        vueron::scatter(pfe_output, voxel_coords, voxel_num_points, num_pillars,
                        bev_image);

        // read bev_features from snapshot file
        const std::string rpn_input_path = snapshot_dir + "/bev_features.npy";
        auto raw_bev_features = npy::read_npy<float>(rpn_input_path);
        std::vector<float> rpn_input_snapshot = raw_bev_features.data;
        EXPECT_EQ(rpn_input_snapshot.size(), bev_image.size());

        for (size_t elem = 0; elem < bev_image.size(); elem++) {
            EXPECT_FLOAT_EQ(bev_image[elem], rpn_input_snapshot[elem]);
            if (bev_image[elem] != rpn_input_snapshot[elem]) {
                int a = 1;
            }
        }
    }
}
