#include "npy.h"
#include "params.h"
#include "pcl.h"
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
        std::vector<float> pfe_output(MAX_VOXELS * RPN_INPUT_NUM_CHANNELS,
                                      0.0f); // input of scatter()
        std::vector<float> bev_image(GRID_Y_SIZE * GRID_X_SIZE *
                                         RPN_INPUT_NUM_CHANNELS,
                                     0.0f); // input of RPN
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
        EXPECT_EQ(pfe_output.size(), MAX_VOXELS * RPN_INPUT_NUM_CHANNELS);

        vueron::scatter(pfe_output, voxel_coords, num_pillars, bev_image);

        // read bev_features from snapshot file
        const std::string rpn_input_path = snapshot_dir + "/bev_features.npy";
        auto raw_bev_features = npy::read_npy<float>(rpn_input_path);
        std::vector<float> rpn_input_snapshot = raw_bev_features.data;
        EXPECT_EQ(rpn_input_snapshot.size(), bev_image.size());

        for (size_t elem = 0; elem < bev_image.size(); elem++) {
            size_t grid_x = elem % GRID_X_SIZE;
            size_t tmp = (elem - grid_x) / GRID_X_SIZE;
            size_t grid_y = tmp % GRID_Y_SIZE;
            size_t feat_idx = tmp / GRID_Y_SIZE;
            assert(elem == (feat_idx * GRID_X_SIZE * GRID_Y_SIZE) +
                               (grid_y * GRID_X_SIZE) + grid_x);
            EXPECT_NEAR(bev_image[elem], rpn_input_snapshot[elem], _EPSILON);
        }
        vueron::rpn_run(bev_image);

        std::cout << "Test Finish : " << pcd_file << std::endl;
    }
}
