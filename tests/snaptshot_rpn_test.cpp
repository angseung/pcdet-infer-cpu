#include "npy.h"
#include "params.h"
#include "pcl.h"
#include "post.h"
#include "pre.h"
#include "rpn.h"
#include "utils.h"
#include <cmath>
#include <glob.h>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

TEST(RPNTest, RPNShapeTest) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files =
        vueron::getFileList(snapshot_folder_path);
    std::vector<float> points;
    size_t num_test_files = pcd_files.size();

    EXPECT_LE(pcd_files.size(), snapshot_files.size());

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
        std::vector<float> pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR *
                                         FEATURE_NUM,
                                     0.0f); // input of pfe_run()
        std::vector<float> pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER,
                                      0.0f); // input of scatter()
        std::vector<float> bev_image(GRID_Y_SIZE * GRID_X_SIZE *
                                         NUM_FEATURE_SCATTER,
                                     0.0f); // input of RPN
        std::vector<std::vector<float>> rpn_output;
        vueron::voxelization(bev_pillar, (float *)points.data(), points_buf_len,
                             point_stride);
        size_t num_pillars = vueron::point_decoration(
            bev_pillar, voxel_coords, voxel_num_points, pfe_input,
            (float *)points.data(), points_buf_len, point_stride);
        size_t num_voxels_manual = std::accumulate(voxel_num_points.begin(),
                                                   voxel_num_points.end(), 0);
        EXPECT_EQ(num_pillars, voxel_num_points.size());

        // check remainder voxels is zero
        float sum_of_pfe_input_remainder = std::accumulate(
            pfe_input.begin() +
                num_pillars * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM,
            pfe_input.end(), 0.0f);
        EXPECT_FLOAT_EQ(sum_of_pfe_input_remainder, 0.0f);

        float remainder_sum = std::accumulate(
            pfe_input.begin() +
                (num_pillars - 1) * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM,
            pfe_input.end(), 0.0f);
        EXPECT_FALSE(remainder_sum == 0.0f);

        vueron::pfe_run(pfe_input, pfe_output);
        EXPECT_EQ(pfe_output.size(), MAX_VOXELS * NUM_FEATURE_SCATTER);

        vueron::scatter(pfe_output, voxel_coords, num_pillars, bev_image);

        // read bev_features from snapshot file
        const std::string rpn_input_path = snapshot_dir + "/bev_features.npy";
        auto raw_bev_features = npy::read_npy<float>(rpn_input_path);
        std::vector<float> rpn_input_snapshot = raw_bev_features.data;
        EXPECT_EQ(rpn_input_snapshot.size(), bev_image.size());

        vueron::rpn_run(bev_image, rpn_output);
        std::vector<size_t> head_output_channels = {
            CLASS_NUM, 3, 2, 1, 2, 1}; // {hm, dim, center, center_z, rot, iou}
        size_t head_dim = GRID_X_SIZE * GRID_Y_SIZE / 4;

        for (size_t j = 0; j < rpn_output.size(); j++) {
            size_t expected_size = head_dim * head_output_channels[j];
            std::vector<float> curr_head_output = rpn_output[j];

            EXPECT_EQ(expected_size, curr_head_output.size());
        }

        vueron::rectify_score(rpn_output[0], rpn_output[5]);

        std::cout << "Test Finish : " << pcd_file << std::endl;
    }
}
