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
    std::vector<float> points;
    size_t num_test_files = pcd_files.size();

    size_t point_stride = POINT_STRIDE;
    std::string pfe_path(PFE_PATH);
    std::string rpn_path(RPN_PATH);
    vueron::PCDet pcdet(pfe_path, rpn_path);

    for (size_t i = 0; i < num_test_files; i++) {
        std::string pcd_file = pcd_files[i];
        std::cout << "Testing : " << pcd_file << std::endl;
        std::vector<float> buffer =
            vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);
        float *points = (float *)buffer.data();
        size_t point_buf_len = buffer.size();

        /*
            Buffers for Inference
        */
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
        std::vector<std::vector<float>> rpn_outputs;
        std::vector<vueron::BndBox> pre_boxes; // boxes before NMS
        std::vector<size_t> pre_labels;        // labels before NMS
        std::vector<float> pre_scores;         // scores before NMS

        // Buffers for inferece for pcdetfunc
        std::vector<vueron::BndBox> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<size_t> nms_labels;

        // Do inference with pcdetfunc
        vueron::run_model(points, point_buf_len, point_stride, nms_boxes,
                          nms_labels, nms_scores);

        // Buffers for inferece for pcdet
        std::vector<vueron::PredBox> pcdet_nms_boxes;

        // Do inference with pcdet
        pcdet.do_infer(points, point_buf_len, point_stride, pcdet_nms_boxes);

        EXPECT_EQ(nms_boxes.size(), MAX_BOX_NUM_AFTER_NMS);
        EXPECT_EQ(nms_scores.size(), MAX_BOX_NUM_AFTER_NMS);
        EXPECT_EQ(nms_labels.size(), MAX_BOX_NUM_AFTER_NMS);
        EXPECT_EQ(pcdet_nms_boxes.size(), MAX_BOX_NUM_AFTER_NMS);

        for (size_t j = 0; j < MAX_BOX_NUM_AFTER_NMS; j++) {
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
