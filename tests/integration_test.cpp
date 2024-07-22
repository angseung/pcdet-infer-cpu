#include <glob.h>
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "pcdet-infer-cpu/common/type.h"
#include "pcdet_test/include/config.h"
#include "pcdet_test/include/model.h"
#include "pcdet_test/include/npy.h"
#include "pcdet_test/include/params.h"
#include "pcl.h"

#define ERROR 1e-3

TEST(IntegrationTest, IntegrationTest) {
  const std::string folder_path = PCD_PATH;
  std::vector<std::string> pcd_files = vueron::getPCDFileList(folder_path);
  const std::string snapshot_folder_path = SNAPSHOT_PATH;
  std::vector<std::string> snapshot_files =
      vueron::getFileList(snapshot_folder_path);
  const size_t num_test_files = pcd_files.size();
  EXPECT_LE(pcd_files.size(), snapshot_files.size());

  for (size_t i = 0; i < num_test_files; i++) {
    const std::string pcd_file = pcd_files[i];
    const std::string snapshot_dir = snapshot_files[i];
    std::cout << "Testing : " << pcd_file << std::endl;
    vueron::PCDReader reader(pcd_file);
    const std::vector<float> buffer = reader.getData();
    const size_t point_stride = reader.getStride();
    const size_t point_buf_len = buffer.size();
    const float *points = buffer.data();

    /*
        Read bev_features from snapshot file
    */
    // boxes
    const std::string boxes_path = snapshot_dir + "/final_boxes.npy";
    auto raw_boxes = npy::read_npy<float>(boxes_path);
    std::vector<float> boxes_snapshot = raw_boxes.data;

    // scores
    const std::string scores_path = snapshot_dir + "/final_scores.npy";
    auto raw_scores = npy::read_npy<float>(scores_path);
    std::vector<float> scores_snapshot = raw_scores.data;

    // labels
    const std::string labels_path = snapshot_dir + "/final_labels.npy";
    auto raw_labels = npy::read_npy<uint32_t>(labels_path);
    std::vector<uint32_t> labels_snapshot = raw_labels.data;

    EXPECT_EQ(boxes_snapshot.size(), 7 * labels_snapshot.size());
    EXPECT_EQ(scores_snapshot.size(), labels_snapshot.size());

    EXPECT_EQ(boxes_snapshot.size(), MAX_OBJ_PER_SAMPLE);
    EXPECT_EQ(scores_snapshot.size(), MAX_OBJ_PER_SAMPLE);
    EXPECT_EQ(labels_snapshot.size(), MAX_OBJ_PER_SAMPLE);

    /*
        Inference with pcd file
    */
    // Buffers for inferece
    std::vector<BndBox> nms_boxes;
    std::vector<float> nms_scores;
    std::vector<size_t> nms_labels;

    // Do inference
    vueron::run_model(points, point_buf_len, point_stride, nms_boxes,
                      nms_labels, nms_scores);

    EXPECT_EQ(nms_boxes.size(), MAX_OBJ_PER_SAMPLE);
    EXPECT_EQ(nms_scores.size(), MAX_OBJ_PER_SAMPLE);
    EXPECT_EQ(nms_labels.size(), MAX_OBJ_PER_SAMPLE);

    for (size_t j = 0; j < labels_snapshot.size(); j++) {
      EXPECT_NEAR(nms_boxes[j].x, boxes_snapshot[7 * j], ERROR);
      EXPECT_NEAR(nms_boxes[j].y, boxes_snapshot[7 * j + 1], ERROR);
      EXPECT_NEAR(nms_boxes[j].z, boxes_snapshot[7 * j + 2], ERROR);
      EXPECT_NEAR(nms_boxes[j].dx, boxes_snapshot[7 * j + 3], ERROR);
      EXPECT_NEAR(nms_boxes[j].dy, boxes_snapshot[7 * j + 4], ERROR);
      EXPECT_NEAR(nms_boxes[j].dz, boxes_snapshot[7 * j + 5], ERROR);
      EXPECT_NEAR(nms_boxes[j].heading, boxes_snapshot[7 * j + 6], ERROR);

      EXPECT_NEAR(scores_snapshot[j], nms_scores[j], ERROR);
      EXPECT_EQ(labels_snapshot[j], nms_labels[j]);
    }

    std::cout << "Test Finish : " << pcd_file << std::endl;
  }
}
