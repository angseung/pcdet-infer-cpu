#include "config.h"
#include "npy.h"
#include "params.h"
#include "pcdet-infer-cpu/model.h"
#include "pcdet-infer-cpu/pcdet.h"
#include "pcl.h"
#include "type.h"
#include <cmath>
#include <glob.h>
#include <gtest/gtest.h>
#include <numeric>

#define _ERROR 1e-3

TEST(IntegrationTest, IntegrationTest) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    std::vector<std::string> snapshot_files =
        vueron::getFileList(snapshot_folder_path);
    std::vector<float> points;
    size_t num_test_files = pcd_files.size();

    size_t point_stride = POINT_STRIDE;

    for (size_t i = 0; i < num_test_files; i++) {
        std::string pcd_file = pcd_files[i];
        std::cout << "Testing : " << pcd_file << std::endl;
        std::vector<float> buffer =
            vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);
        float *points = (float *)buffer.data();
        size_t point_buf_len = buffer.size();

        /*
            Inference with pcd file
        */
        // Buffers for inferece
        std::vector<vueron::BndBox> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<size_t> nms_labels;

        // Do inference
        vueron::run_model(points, point_buf_len, point_stride, nms_boxes,
                          nms_scores, nms_labels);

        EXPECT_EQ(nms_boxes.size(), MAX_BOX_NUM_AFTER_NMS);
        EXPECT_EQ(nms_scores.size(), MAX_BOX_NUM_AFTER_NMS);
        EXPECT_EQ(nms_labels.size(), MAX_BOX_NUM_AFTER_NMS);

        for (size_t j = 0; j < labels_snapshot.size(); j++) {
            EXPECT_NEAR(nms_boxes[j].x, boxes_snapshot[7 * j], _ERROR);
            EXPECT_NEAR(nms_boxes[j].y, boxes_snapshot[7 * j + 1], _ERROR);
            EXPECT_NEAR(nms_boxes[j].z, boxes_snapshot[7 * j + 2], _ERROR);
            EXPECT_NEAR(nms_boxes[j].dx, boxes_snapshot[7 * j + 3], _ERROR);
            EXPECT_NEAR(nms_boxes[j].dy, boxes_snapshot[7 * j + 4], _ERROR);
            EXPECT_NEAR(nms_boxes[j].dz, boxes_snapshot[7 * j + 5], _ERROR);
            EXPECT_NEAR(nms_boxes[j].heading, boxes_snapshot[7 * j + 6],
                        _ERROR);

            EXPECT_NEAR(scores_snapshot[j], nms_scores[j], _ERROR);
            EXPECT_EQ(labels_snapshot[j], nms_labels[j]);
        }

        std::cout << "Test Finish : " << pcd_file << std::endl;
    }
}
