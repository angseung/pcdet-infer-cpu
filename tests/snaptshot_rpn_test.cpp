#include <glob.h>
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "pcdet-infer-cpu/common/box.h"
#include "pcdet_test/include/config.h"
#include "pcdet_test/include/npy.h"
#include "pcdet_test/include/params.h"
#include "pcdet_test/include/post.h"
#include "pcdet_test/include/pre.h"
#include "pcdet_test/include/rpn.h"
#include "pcl.h"

#define _EPSILON_RPN 5e-3
#define _EPSILON_HM 1e-3
#define _EPSILON_DIM 1e-3

TEST(RPNTest, RPNShapeTest) {
  std::string folder_path = PCD_PATH;
  std::vector<std::string> pcd_files = vueron::getPCDFileList(folder_path);
  std::string snapshot_folder_path = SNAPSHOT_PATH;
  std::vector<std::string> snapshot_files =
      vueron::getFileList(snapshot_folder_path);
  size_t num_test_files = pcd_files.size();

  EXPECT_LE(pcd_files.size(), snapshot_files.size());

  for (size_t i = 0; i < num_test_files; i++) {
    std::string pcd_file = pcd_files[i];
    std::string snapshot_dir = snapshot_files[i];
    std::cout << "Testing : " << pcd_file << std::endl;

    // read bev_features from snapshot file
    const std::string rpn_input_path = snapshot_dir + "/bev_features.npy";
    auto raw_bev_features = npy::read_npy<float>(rpn_input_path);
    std::vector<float> rpn_input_snapshot = raw_bev_features.data;

    std::vector<std::vector<float>> rpn_output;
    vueron::rpn_run(rpn_input_snapshot, rpn_output);
    std::vector<size_t> head_output_channels{
        static_cast<size_t>(CLASS_NUM),
        3,
        2,
        1,
        2,
        1};  // {hm, dim, center, center_z, rot, iou}
    size_t head_dim = GRID_X_SIZE * GRID_Y_SIZE / 4;

    for (size_t j = 0; j < rpn_output.size(); j++) {
      size_t expected_size = head_dim * head_output_channels[j];
      std::vector<float> curr_head_output = rpn_output[j];

      EXPECT_EQ(expected_size, curr_head_output.size());
    }

    std::cout << "Test Finish : " << pcd_file << std::endl;
  }
}

TEST(RPNTest, RPNValueTest) {
  std::string folder_path = PCD_PATH;
  std::vector<std::string> pcd_files = vueron::getPCDFileList(folder_path);
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
    vueron::PCDReader reader(pcd_file);
    const std::vector<float> &points = reader.getData();
    const size_t point_stride = reader.getStride();
    const size_t point_buf_len = points.size();
    std::vector<vueron::Pillar> bev_pillar(
        GRID_Y_SIZE * GRID_X_SIZE, vueron::Pillar(MAX_NUM_POINTS_PER_PILLAR));
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
    std::vector<std::vector<float>> rpn_output;
    voxelization(bev_pillar, points.data(), point_buf_len, point_stride);
    size_t num_pillars =
        point_decoration(bev_pillar, voxel_coords, voxel_num_points, pfe_input,
                         points.data(), point_stride);
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
    std::vector<size_t> head_output_channels{
        static_cast<size_t>(CLASS_NUM),  // hm
        3,                               // dim
        2,                               // center
        1,                               // center_z
        2,                               // rot
        1                                // iou
    };

    size_t head_dim = GRID_X_SIZE * GRID_Y_SIZE / 4;

    // read snapshot file
    // 1. center
    const std::string center_path = snapshot_dir + "/center.npy";
    auto raw_center = npy::read_npy<float>(center_path);
    std::vector<float> center_snapshot = raw_center.data;
    EXPECT_EQ(head_dim * 2, center_snapshot.size());

    for (size_t j = 0; j < center_snapshot.size(); j++) {
      EXPECT_NEAR(center_snapshot[j], rpn_output[2][j], _EPSILON_RPN);
    }

    // 2. center_z
    const std::string center_z_path = snapshot_dir + "/center_z.npy";
    auto raw_center_z = npy::read_npy<float>(center_z_path);
    std::vector<float> center_z_snapshot = raw_center_z.data;
    EXPECT_EQ(head_dim, center_z_snapshot.size());

    for (size_t j = 0; j < center_z_snapshot.size(); j++) {
      EXPECT_NEAR(center_z_snapshot[j], rpn_output[3][j], _EPSILON_RPN);
    }

    // 3. dim
    const std::string dim_path = snapshot_dir + "/dim.npy";
    auto raw_dim = npy::read_npy<float>(dim_path);
    std::vector<float> dim_snapshot = raw_dim.data;
    EXPECT_EQ(head_dim * 3, dim_snapshot.size());

    for (size_t j = 0; j < dim_snapshot.size(); j++) {
      EXPECT_NEAR(dim_snapshot[j], rpn_output[1][j], _EPSILON_DIM);
    }

    // 4. rot
    const std::string rot_path = snapshot_dir + "/rot.npy";
    auto raw_rot = npy::read_npy<float>(rot_path);
    std::vector<float> rot_snapshot = raw_rot.data;
    EXPECT_EQ(head_dim * 2, rot_snapshot.size());

    for (size_t j = 0; j < rot_snapshot.size(); j++) {
      EXPECT_NEAR(rot_snapshot[j], rpn_output[4][j], _EPSILON_RPN);
    }

    // 5. iou
    const std::string iou_path = snapshot_dir + "/iou.npy";
    auto raw_iou = npy::read_npy<float>(iou_path);
    std::vector<float> iou_snapshot = raw_iou.data;
    EXPECT_EQ(head_dim, iou_snapshot.size());

    for (size_t j = 0; j < iou_snapshot.size(); j++) {
      EXPECT_NEAR(iou_snapshot[j], rpn_output[5][j], _EPSILON_RPN);
    }

    // 6. hm
    const std::string hm_path = snapshot_dir + "/hm.npy";
    auto raw_hm = npy::read_npy<float>(hm_path);
    std::vector<float> hm_snapshot = raw_hm.data;
    EXPECT_EQ(head_dim * CLASS_NUM, hm_snapshot.size());

    for (size_t j = 0; j < hm_snapshot.size(); j++) {
      EXPECT_NEAR(hm_snapshot[j], rpn_output[0][j], _EPSILON_HM);
    }

    std::cout << "Test Finish : " << pcd_file << std::endl;
  }
}
