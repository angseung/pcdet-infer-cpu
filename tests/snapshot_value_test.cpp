#include <glob.h>
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "config.h"
#include "npy.h"
#include "params.h"
#include "pcdet-infer-cpu/pre.h"
#include "pcdet-infer-cpu/rpn.h"
#include "pcl.h"

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
    const std::vector<float> points =
        vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);
    const size_t points_buf_len = points.size();
    constexpr size_t point_stride = POINT_STRIDE;
    std::vector<float> bev_feature(
        GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER,
        0.0f);  // input of RPN

    // read snapshot file
    // 1. pfe_output
    const std::string pfe_output_path = snapshot_dir + "/padded_pfe_output.npy";
    auto raw_pfe_output = npy::read_npy<float>(pfe_output_path);
    std::vector<float> pfe_output_snapshot = raw_pfe_output.data;

    // 2. voxel_coord
    const std::string voxel_coord_path =
        snapshot_dir + "/padded_voxel_coord.npy";
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
        snapshot_dir + "/padded_voxel_num_points.npy";
    auto raw_voxel_num_points = npy::read_npy<uint32_t>(voxel_num_points_path);
    std::vector<uint32_t> voxel_num_points_snapshot = raw_voxel_num_points.data;

    // 4. bev_features
    const std::string rpn_input_path = snapshot_dir + "/bev_features.npy";
    auto raw_bev_features = npy::read_npy<float>(rpn_input_path);
    std::vector<float> rpn_input_snapshot = raw_bev_features.data;

    vueron::scatter(pfe_output_snapshot, voxel_coords,
                    voxel_num_points_snapshot.size(), bev_feature);

    for (size_t j = 0; j < bev_feature.size(); j++) {
      EXPECT_FLOAT_EQ(rpn_input_snapshot[j], bev_feature[j]);
    }
    std::cout << "Test Finish : " << pcd_file << std::endl;
  }
}

TEST(VoxelValueTest, PFERunTest) {
  std::string folder_path = PCD_PATH;
  std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
  std::string snapshot_folder_path = SNAPSHOT_PATH;
  std::vector<std::string> snapshot_files =
      vueron::getFileList(snapshot_folder_path);
  size_t num_test_files = pcd_files.size();

  EXPECT_LE(pcd_files.size(), snapshot_files.size());

  for (size_t i = 0; i < num_test_files; i++) {
    std::string pcd_file = pcd_files[i];
    std::string snapshot_dir = snapshot_files[i];
    std::cout << "Testing : " << pcd_file << std::endl;

    // read point from pcd file
    const std::vector<float> points =
        vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);
    const size_t points_buf_len = points.size();
    constexpr size_t point_stride = POINT_STRIDE;
    std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE,
                                           MAX_NUM_POINTS_PER_PILLAR);
    std::vector<size_t> voxel_coords;  // (x, y)
    std::vector<size_t> voxel_num_points;
    std::vector<float> pfe_input(
        MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM,
        0.0f);  // input of pfe_run()
    std::vector<float> bev_feature(
        GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER,
        0.0f);  // input of RPN

    voxelization(bev_pillar, (float *)points.data(), points_buf_len,
                 point_stride);
    size_t num_pillars =
        point_decoration(bev_pillar, voxel_coords, voxel_num_points, pfe_input,
                         (float *)points.data(), point_stride);

    // read snapshot file
    // 1. pfe_input
    const std::string pfe_input_path =
        snapshot_dir + "/padded_voxels_encoded.npy";
    auto raw_pfe_input = npy::read_npy<float>(pfe_input_path);
    std::vector<float> pfe_input_snapshot = raw_pfe_input.data;

    // 2. pfe_output
    const std::string pfe_output_path = snapshot_dir + "/padded_pfe_output.npy";
    auto raw_pfe_output = npy::read_npy<float>(pfe_output_path);
    std::vector<float> pfe_output_snapshot = raw_pfe_output.data;
    std::vector<float> pfe_output{MAX_VOXELS * NUM_FEATURE_SCATTER, 0.0f};

    vueron::pfe_run(pfe_input_snapshot, pfe_output);
    for (size_t j = 0; j < num_pillars * NUM_FEATURE_SCATTER; j++) {
      EXPECT_NEAR(pfe_output[j], pfe_output_snapshot[j], _EPSILON);
    }
    std::cout << "Test Finish : " << pcd_file << std::endl;
  }
}

TEST(VoxelValueTest, BEVValueTest) {
  std::string folder_path = PCD_PATH;
  std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
  std::string snapshot_folder_path = SNAPSHOT_PATH;
  std::vector<std::string> snapshot_files =
      vueron::getFileList(snapshot_folder_path);
  size_t num_test_files = pcd_files.size();

  EXPECT_LE(pcd_files.size(), snapshot_files.size());

  for (size_t i = 0; i < num_test_files; i++) {
    std::string pcd_file = pcd_files[i];
    std::string snapshot_dir = snapshot_files[i];
    std::cout << "Testing : " << pcd_file << std::endl;

    // read point from pcd file
    const std::vector<float> points =
        vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);
    const size_t points_buf_len = points.size();
    constexpr size_t point_stride = POINT_STRIDE;
    std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE,
                                           MAX_NUM_POINTS_PER_PILLAR);
    std::vector<size_t> voxel_coords;  // (x, y)
    std::vector<size_t> voxel_num_points;
    std::vector<float> pfe_input(
        MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM,
        0.0f);  // input of pfe_run()
    std::vector<float> pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER,
                                  0.0f);  // input of scatter()
    std::vector<float> bev_image(
        GRID_Y_SIZE * GRID_X_SIZE * NUM_FEATURE_SCATTER,
        0.0f);  // input of RPN
    vueron::voxelization(bev_pillar, (float *)points.data(), points_buf_len,
                         point_stride);
    size_t num_pillars = vueron::point_decoration(
        bev_pillar, voxel_coords, voxel_num_points, pfe_input,
        (float *)points.data(), point_stride);
    size_t num_voxels_manual =
        std::accumulate(voxel_num_points.begin(), voxel_num_points.end(), 0);
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

    for (size_t elem = 0; elem < bev_image.size(); elem++) {
      size_t grid_x = elem % GRID_X_SIZE;
      size_t tmp = (elem - grid_x) / GRID_X_SIZE;
      size_t grid_y = tmp % GRID_Y_SIZE;
      size_t feat_idx = tmp / GRID_Y_SIZE;
      assert(elem == (feat_idx * GRID_X_SIZE * GRID_Y_SIZE) +
                         (grid_y * GRID_X_SIZE) + grid_x);
      EXPECT_NEAR(bev_image[elem], rpn_input_snapshot[elem], _EPSILON);
    }

    std::cout << "Test Finish : " << pcd_file << std::endl;
  }
}
