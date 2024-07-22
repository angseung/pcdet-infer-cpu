#include <glob.h>
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "pcdet-infer-cpu/common/type.h"
#include "pcdet_test/include/config.h"
#include "pcdet_test/include/model.h"
#include "pcdet_test/include/npy.h"
#include "pcdet_test/include/params.h"
#include "pcdet_test/include/pcdet.h"
#include "pcl.h"

TEST(IntegrationTest, IntegrationTest) {
  std::string folder_path = PCD_PATH;
  std::vector<std::string> pcd_files = vueron::getPCDFileList(folder_path);
  const size_t num_test_files = pcd_files.size();
  std::string pfe_path(PFE_FILE);
  std::string rpn_path(RPN_FILE);
  const auto pcdet = std::make_unique<vueron::PCDetCPU>(pfe_path, rpn_path);

  for (size_t i = 0; i < num_test_files; i++) {
    const std::string pcd_file = pcd_files[i];
    std::cout << "Testing : " << pcd_file << std::endl;
    vueron::PCDReader reader(pcd_file);
    const std::vector<float> buffer = reader.getData();
    const size_t point_stride = reader.getStride();
    const size_t point_buf_len = buffer.size();
    const float *points = buffer.data();

    /*
        Buffers for Inference
    */
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
    std::vector<std::vector<float>> rpn_outputs;
    std::vector<BndBox> pre_boxes;   // boxes before NMS
    std::vector<size_t> pre_labels;  // labels before NMS
    std::vector<float> pre_scores;   // scores before NMS

    // Buffers for inferece for pcdetfunc
    std::vector<BndBox> nms_boxes;
    std::vector<size_t> nms_labels;
    std::vector<float> nms_scores;

    // Do inference with pcdetfunc
    vueron::run_model(points, point_buf_len, point_stride, nms_boxes,
                      nms_labels, nms_scores);

    // Buffers for inferece for pcdet
    std::vector<PredBox> pcdet_nms_boxes;

    // Do inference with pcdet
    pcdet->run(points, point_buf_len, point_stride, pcdet_nms_boxes);

    EXPECT_EQ(nms_boxes.size(), MAX_OBJ_PER_SAMPLE);
    EXPECT_EQ(nms_scores.size(), MAX_OBJ_PER_SAMPLE);
    EXPECT_EQ(nms_labels.size(), MAX_OBJ_PER_SAMPLE);
    EXPECT_EQ(pcdet_nms_boxes.size(), MAX_OBJ_PER_SAMPLE);

    for (size_t j = 0; j < MAX_OBJ_PER_SAMPLE; j++) {
      EXPECT_EQ(nms_labels[j], pcdet_nms_boxes[j].label);
      EXPECT_FLOAT_EQ(nms_scores[j], pcdet_nms_boxes[j].score);
      EXPECT_FLOAT_EQ(nms_boxes[j].x, pcdet_nms_boxes[j].x);
      EXPECT_FLOAT_EQ(nms_boxes[j].y, pcdet_nms_boxes[j].y);
      EXPECT_FLOAT_EQ(nms_boxes[j].z, pcdet_nms_boxes[j].z);
      EXPECT_FLOAT_EQ(nms_boxes[j].dx, pcdet_nms_boxes[j].dx);
      EXPECT_FLOAT_EQ(nms_boxes[j].dy, pcdet_nms_boxes[j].dy);
      EXPECT_FLOAT_EQ(nms_boxes[j].dz, pcdet_nms_boxes[j].dz);
      EXPECT_FLOAT_EQ(nms_boxes[j].heading, pcdet_nms_boxes[j].heading);
    }

    std::cout << "Test Finish : " << pcd_file << std::endl;
  }
}
