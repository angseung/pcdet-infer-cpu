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

TEST(VoxelValueTest, ScatterTest) {
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
        std::vector<float> bev_feature(GRID_Y_SIZE * GRID_X_SIZE *
                                           RPN_INPUT_NUM_CHANNELS,
                                       0.0f); // input of RPN

        // read snapshot file
        // 1. pfe_output
        const std::string pfe_output_path = snapshot_dir + "/pfe_output.npy";
        auto raw_pfe_output = npy::read_npy<float>(pfe_output_path);
        std::vector<float> pfe_output_snapshot = raw_pfe_output.data;

        // 2. voxel_coord
        const std::string voxel_coord_path = snapshot_dir + "/voxel_coord.npy";
        auto raw_voxel_coord = npy::read_npy<uint32_t>(voxel_coord_path);
        std::vector<uint32_t> voxel_coord_snapshot = raw_voxel_coord.data;
        std::vector<size_t> voxel_coords;

        for (size_t j = 0; j < voxel_coord_snapshot.size() / 2; j++) {
            voxel_coords.push_back((size_t)voxel_coord_snapshot[2 * j + 1]);
            voxel_coords.push_back((size_t)voxel_coord_snapshot[2 * j]);
        }

        EXPECT_EQ(voxel_coord_snapshot.size(), voxel_coords.size());

        // 3. voxel_num_points
        const std::string voxel_num_points_path =
            snapshot_dir + "/voxel_num_points.npy";
        auto raw_voxel_num_points =
            npy::read_npy<uint32_t>(voxel_num_points_path);
        std::vector<uint32_t> voxel_num_points_snapshot =
            raw_voxel_num_points.data;

        // 4. bev_features
        const std::string rpn_input_path = snapshot_dir + "/bev_features.npy";
        auto raw_bev_features = npy::read_npy<float>(rpn_input_path);
        std::vector<float> rpn_input_snapshot = raw_bev_features.data;

        vueron::scatter(pfe_output_snapshot, voxel_coords,
                        voxel_num_points_snapshot.size(), bev_feature);

        const std::vector<unsigned long> leshape12{RPN_INPUT_NUM_CHANNELS,
                                                   GRID_Y_SIZE, GRID_X_SIZE};
        const npy::npy_data<float> data12{bev_feature, leshape12, false};
        write_npy(snapshot_dir + "/bev_feature_CXX.npy", data12);

        for (size_t j = 0; j < bev_feature.size(); j++) {
            EXPECT_FLOAT_EQ(rpn_input_snapshot[j], bev_feature[j]);
        }
    }
}

TEST(VoxelValueTest, PFERunTest) {
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
        std::vector<float> bev_feature(GRID_Y_SIZE * GRID_X_SIZE *
                                           RPN_INPUT_NUM_CHANNELS,
                                       0.0f); // input of RPN

        // read snapshot file
        // 1. pfe_input
        const std::string pfe_input_path = snapshot_dir + "/voxels_encoded.npy";
        auto raw_pfe_input = npy::read_npy<float>(pfe_input_path);
        std::vector<float> pfe_input_snapshot = raw_pfe_input.data;

        // 2. pfe_output
        const std::string pfe_output_path = snapshot_dir + "/pfe_output.npy";
        auto raw_pfe_output = npy::read_npy<float>(pfe_output_path);
        std::vector<float> pfe_output_snapshot = raw_pfe_output.data;
        std::vector<float> pfe_output{MAX_VOXELS * RPN_INPUT_NUM_CHANNELS,
                                      0.0f};

        vueron::run(pfe_input_snapshot, pfe_output);

        const std::vector<unsigned long> leshape12{MAX_VOXELS *
                                                   RPN_INPUT_NUM_CHANNELS};
        const npy::npy_data<float> data12{pfe_output, leshape12, false};
        write_npy(snapshot_dir + "/pfe_output_CXX.npy", data12);

        for (size_t j = 0; j < pfe_output.size(); j++) {
            EXPECT_FLOAT_EQ(pfe_output[j], pfe_output_snapshot[j]);
        }
    }
}

// TEST(VoxelSnapshotTest, BEVValueTest) {
//     std::string folder_path = PCD_PATH;
//     std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
//     std::string snapshot_folder_path = SNAPSHOT_PATH;
//     std::vector<std::string> snapshot_files =
//         vueron::getFileList(snapshot_folder_path);
//     std::vector<float> points;
//     size_t num_test_files = pcd_files.size();

//     EXPECT_LE(pcd_files.size(), snapshot_files.size());

//     for (size_t i = 0; i < num_test_files; i++) {
//         std::string pcd_file = pcd_files[i];
//         std::string snapshot_dir = snapshot_files[i];
//         std::cout << "Testing : " << pcd_file << std::endl;

//         // read point from pcd file
//         std::vector<float> points =
//             vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);
//         size_t points_buf_len = points.size();
//         size_t point_stride = sizeof(float);
//         std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
//         std::vector<size_t> voxel_coords; // (x, y)
//         std::vector<size_t> voxel_num_points;
//         std::vector<vueron::Voxel> raw_voxels(
//             GRID_Y_SIZE * GRID_X_SIZE *
//             MAX_NUM_POINTS_PER_PILLAR); // input of gather()
//         std::vector<float> pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR *
//                                          FEATURE_NUM,
//                                      0.0f); // input of run()
//         std::vector<float> pfe_output(MAX_VOXELS * RPN_INPUT_NUM_CHANNELS,
//                                       0.0f); // input of scatter()
//         std::vector<float> bev_image(GRID_Y_SIZE * GRID_X_SIZE *
//                                          RPN_INPUT_NUM_CHANNELS,
//                                      0.0f); // input of RPN
//         vueron::voxelization(bev_pillar, (float *)points.data(),
//         points_buf_len,
//                              point_stride);
//         size_t num_pillars = vueron::point_decoration(
//             bev_pillar, voxel_coords, voxel_num_points, raw_voxels,
//             (float *)points.data(), points_buf_len, point_stride);
//         size_t num_valid_voxels = vueron::gather(raw_voxels, pfe_input);
//         size_t num_voxels_manual = std::accumulate(voxel_num_points.begin(),
//                                                    voxel_num_points.end(),
//                                                    0);
//         EXPECT_EQ(num_voxels_manual, num_valid_voxels);
//         EXPECT_EQ(num_pillars, voxel_num_points.size());

//         // for (size_t j = 0; j < voxel_num_points.size(); j++) {
//         //     size_t curr_num_points = voxel_num_points[j];
//         // }
//         vueron::run(pfe_input, pfe_output);
//         EXPECT_EQ(pfe_output.size(), MAX_VOXELS * RPN_INPUT_NUM_CHANNELS);

//         const std::vector<unsigned long> leshape11{
//             MAX_VOXELS, MAX_NUM_POINTS_PER_PILLAR, FEATURE_NUM};
//         const npy::npy_data<float> data11{pfe_input, leshape11, false};
//         write_npy(snapshot_dir + "/pfe_input_CXX.npy", data11);

//         const std::vector<unsigned long> leshape12{MAX_VOXELS,
//                                                    RPN_INPUT_NUM_CHANNELS};
//         const npy::npy_data<float> data12{pfe_output, leshape12, false};
//         write_npy(snapshot_dir + "/pfe_output_CXX.npy", data12);

//         // check remainder voxels is zero
//         size_t padded_voxel_offset = num_valid_voxels * FEATURE_NUM;
//         float remainder_sum = std::accumulate(
//             pfe_input.begin() + padded_voxel_offset, pfe_input.end(), 0.0f);
//         assert(remainder_sum == 0.0f);

//         vueron::scatter(pfe_output, voxel_coords, voxel_num_points,
//         num_pillars,
//                         bev_image);

//         // read bev_features from snapshot file
//         const std::string rpn_input_path = snapshot_dir +
//         "/bev_features.npy"; auto raw_bev_features =
//         npy::read_npy<float>(rpn_input_path); std::vector<float>
//         rpn_input_snapshot = raw_bev_features.data;
//         EXPECT_EQ(rpn_input_snapshot.size(), bev_image.size());

//         // for (size_t elem = 0; elem < bev_image.size(); elem++) {
//         //     EXPECT_FLOAT_EQ(bev_image[elem], rpn_input_snapshot[elem]);
//         //     if (bev_image[elem] != rpn_input_snapshot[elem]) {
//         //         int a = 1;
//         //     }
//         // }
//     }
// }
