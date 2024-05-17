#include "npy.h"
#include "params.h"
#include "pcl.h"
#include "pre.h"
#include "shape.h"
#include <glob.h>
#include <gtest/gtest.h>
#include <iostream>

TEST(VoxelSnapshotTest, VoxelShapeTest) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = getFiles(folder_path);
    std::vector<float> points;

    for (const auto &file : pcd_files) {
        points = readPcdFile(file, MAX_POINTS_NUM);
        vueron::preprocess((float *)points.data(), points.size(),
                           sizeof(float));
    }

    // read snapshot
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files = getFiles(snapshot_folder_path);

    for (std::string snapshot_dir : snapshot_files) {
        const std::string voxels_path = snapshot_dir + "/voxels.npy";
        const std::string voxel_coord_path = snapshot_dir + "/voxel_coord.npy";
        const std::string voxel_num_points_path =
            snapshot_dir + "/voxel_num_points.npy";
        auto raw_voxels = npy::read_npy<float>(voxels_path);
        auto raw_voxel_coord = npy::read_npy<uint32_t>(voxel_coord_path);
        auto raw_voxel_num_points =
            npy::read_npy<uint32_t>(voxel_num_points_path);

        std::vector<float> voxels = raw_voxels.data;
        std::vector<uint32_t> voxel_coord = raw_voxel_coord.data;
        std::vector<uint32_t> voxel_num_points = raw_voxel_num_points.data;

        std::vector<unsigned long> voxel_shape = raw_voxels.shape;
        std::vector<unsigned long> voxel_coord_shape = raw_voxel_coord.shape;
        std::vector<unsigned long> voxel_num_points_shape =
            raw_voxel_num_points.shape;

        // check number of pillars
        EXPECT_EQ(voxel_num_points_shape[0], voxel_shape[0]);
        EXPECT_EQ(voxel_num_points_shape[0], voxel_coord_shape[0]);
        EXPECT_EQ(voxel_shape[0], voxel_coord_shape[0]);

        // check Tensor shape
        EXPECT_EQ(voxels.size(), voxel_shape[0] * MAX_NUM_POINTS_PER_PILLAR *
                                     NUM_POINT_VALUES);
        EXPECT_EQ(voxel_coord.size(), voxel_coord_shape[0] * 2);
    }
}

TEST(VoxelSnapshotTest, VoxelValueTest) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = getFiles(folder_path);
    std::vector<float> points;

    for (const auto &file : pcd_files) {
        points = readPcdFile(file, MAX_POINTS_NUM);
        vueron::preprocess((float *)points.data(), points.size(),
                           sizeof(float));
    }

    // read snapshot
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files = getFiles(snapshot_folder_path);

    for (std::string snapshot_dir : snapshot_files) {
        const std::string voxels_path = snapshot_dir + "/voxels.npy";
        const std::string voxel_coord_path = snapshot_dir + "/voxel_coord.npy";
        const std::string voxel_num_points_path =
            snapshot_dir + "/voxel_num_points.npy";
        auto raw_voxels = npy::read_npy<float>(voxels_path);
        auto raw_voxel_coord = npy::read_npy<uint32_t>(voxel_coord_path);
        auto raw_voxel_num_points =
            npy::read_npy<uint32_t>(voxel_num_points_path);

        std::vector<float> voxels = raw_voxels.data;
        std::vector<uint32_t> voxel_coord = raw_voxel_coord.data;
        std::vector<uint32_t> voxel_num_points = raw_voxel_num_points.data;

        std::vector<unsigned long> voxel_shape = raw_voxels.shape;
        std::vector<unsigned long> voxel_coord_shape = raw_voxel_coord.shape;
        std::vector<unsigned long> voxel_num_points_shape =
            raw_voxel_num_points.shape;

        // check number of pillars
        EXPECT_EQ(voxel_num_points_shape[0], voxel_shape[0]);
        EXPECT_EQ(voxel_num_points_shape[0], voxel_coord_shape[0]);
        EXPECT_EQ(voxel_shape[0], voxel_coord_shape[0]);

        // check Tensor shape
        EXPECT_EQ(voxels.size(), voxel_shape[0] * MAX_NUM_POINTS_PER_PILLAR *
                                     NUM_POINT_VALUES);
        EXPECT_EQ(voxel_coord.size(), voxel_coord_shape[0] * 2);
    }
}
