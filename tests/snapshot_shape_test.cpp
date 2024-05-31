#include "config.h"
#include "npy.h"
#include "params.h"
#include "pcdet-infer-cpu/pre.h"
#include "pcl.h"
#include <glob.h>
#include <gtest/gtest.h>
#include <numeric>

TEST(VoxelSnapshotTest, VoxelShapeTest) {
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
        const std::string bev_features_path =
            snapshot_dir + "/bev_features.npy";
        auto raw_voxels = npy::read_npy<float>(voxels_path);
        auto raw_voxels_encoded = npy::read_npy<float>(voxels_encoded_path);
        auto raw_voxel_coord = npy::read_npy<uint32_t>(voxel_coord_path);
        auto raw_voxel_num_points =
            npy::read_npy<uint32_t>(voxel_num_points_path);
        auto raw_bev_features = npy::read_npy<float>(bev_features_path);

        std::vector<float> voxels = raw_voxels.data;
        std::vector<float> voxels_encoded = raw_voxels_encoded.data;
        std::vector<uint32_t> voxel_coord = raw_voxel_coord.data;
        std::vector<uint32_t> voxel_num_points = raw_voxel_num_points.data;
        std::vector<float> bev_features = raw_bev_features.data;

        std::vector<unsigned long> voxel_shape = raw_voxels.shape;
        std::vector<unsigned long> voxels_encoded_shape =
            raw_voxels_encoded.shape;
        std::vector<unsigned long> voxel_coord_shape = raw_voxel_coord.shape;
        std::vector<unsigned long> voxel_num_points_shape =
            raw_voxel_num_points.shape;
        std::vector<unsigned long> bev_features_shape = raw_bev_features.shape;

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
        EXPECT_EQ(bev_features_shape.size(), 3);
        EXPECT_EQ(bev_features.size(),
                  GRID_X_SIZE * GRID_Y_SIZE * NUM_FEATURE_SCATTER);
    }
}

TEST(VoxelSnapshotTest, PFEShapeTest) {
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
        size_t point_stride = POINT_STRIDE;
        std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
        std::vector<size_t> voxel_coords; // (x, y)
        std::vector<size_t> voxel_num_points;
        std::vector<float> pfe_input(MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR *
                                         FEATURE_NUM,
                                     0.0f); // input of pfe_run()
        std::vector<float> pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER,
                                      0.0f); // input of scatter()
        vueron::pfe_run(pfe_input, pfe_output);
        EXPECT_EQ(pfe_output.size(), MAX_VOXELS * NUM_FEATURE_SCATTER);
        std::cout << "Test Finish : " << pcd_file << std::endl;
    }
}

TEST(VoxelSnapshotTest, VoxelCoordsValueTest) {
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
        points = vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);

        // read voxels from snapshot file
        const std::string voxel_coord_path = snapshot_dir + "/voxel_coord.npy";
        auto raw_voxel_coord = npy::read_npy<uint32_t>(voxel_coord_path);

        // voxel_coords_snapshot : (y, x)
        std::vector<uint32_t> voxel_coords_snapshot = raw_voxel_coord.data;

        // calc values from preprocessing functions
        std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
        std::vector<size_t> voxel_coords; // (x, y)
        std::vector<size_t> voxel_num_points;
        std::vector<float> pfe_input(
            MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM, 0.0f);

        vueron::voxelization(bev_pillar, (float *)points.data(), points.size(),
                             POINT_STRIDE);
        size_t num_pillars = vueron::point_decoration(
            bev_pillar, voxel_coords, voxel_num_points, pfe_input,
            (float *)points.data(), points.size(), POINT_STRIDE);

        EXPECT_EQ(voxel_coords.size() / 2, voxel_num_points.size());
        EXPECT_EQ(voxel_coords.size(), voxel_coords_snapshot.size());

        std::vector<size_t> voxel_coords_x;
        std::vector<size_t> voxel_coords_y;
        std::vector<size_t> voxel_coords_snapshot_x;
        std::vector<size_t> voxel_coords_snapshot_y;

        for (size_t j = 0; j < voxel_coords.size() / 2; j++) {
            voxel_coords_x.push_back(voxel_coords[2 * j]);
            voxel_coords_y.push_back(voxel_coords[2 * j + 1]);
            voxel_coords_snapshot_x.push_back(voxel_coords_snapshot[2 * j + 1]);
            voxel_coords_snapshot_y.push_back(voxel_coords_snapshot[2 * j]);
        }
        EXPECT_EQ(voxel_coords_x.size(), voxel_coords.size() / 2);

        std::sort(voxel_coords_x.begin(), voxel_coords_x.end());
        std::sort(voxel_coords_y.begin(), voxel_coords_y.end());
        std::sort(voxel_coords_snapshot_x.begin(),
                  voxel_coords_snapshot_x.end());
        std::sort(voxel_coords_snapshot_y.begin(),
                  voxel_coords_snapshot_y.end());

        for (size_t j = 0; j < voxel_coords_x.size(); j++) {
            EXPECT_EQ(voxel_coords_x[j], voxel_coords_snapshot_x[j]);
            EXPECT_EQ(voxel_coords_y[j], voxel_coords_snapshot_y[j]);
        }
        std::cout << "Test Finish : " << pcd_file << std::endl;
    }
}
