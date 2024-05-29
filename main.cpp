#include "config.h"
#include "npy.h"
#include "params.h"
#include "pcl.h"
#include "post.h"
#include "pre.h"
#include "rpn.h"
#include "type.h"
#include <glob.h>
#include <iostream>

int main(int argc, const char **argv) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    std::vector<float> points;

    while (1) {
        for (const auto &file : pcd_files) {
            points = vueron::readPcdFile(file, MAX_POINTS_NUM);
#ifdef _DEBUG
            std::cout << file << std::endl;
            std::cout << "Points Num of " << file << ": "
                      << points.size() / sizeof(float) << std::endl;
#endif
            size_t points_buf_len = points.size();
            size_t point_stride = sizeof(float);
            std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
            std::vector<size_t> voxel_coords; // (x, y)
            std::vector<size_t> voxel_num_points;
            std::vector<float> pfe_input(
                MAX_VOXELS * MAX_NUM_POINTS_PER_PILLAR * FEATURE_NUM,
                0.0f); // input of pfe_run()
            std::vector<float> pfe_output(MAX_VOXELS * NUM_FEATURE_SCATTER,
                                          0.0f); // input of scatter()
            std::vector<float> bev_image(GRID_Y_SIZE * GRID_X_SIZE *
                                             NUM_FEATURE_SCATTER,
                                         0.0f);          // input of rpn_run()
            std::vector<std::vector<float>> rpn_outputs; // output of rpn_run()
            std::vector<vueron::BndBox> boxes;           // boxes before NMS
            std::vector<size_t> labels;                  // labels before NMS
            std::vector<float> scores;                   // scores before NMS
            boxes.reserve(MAX_BOX_NUM_BEFORE_NMS);
            labels.reserve(MAX_BOX_NUM_BEFORE_NMS);
            scores.reserve(MAX_BOX_NUM_BEFORE_NMS);

            vueron::voxelization(bev_pillar, (float *)points.data(),
                                 points_buf_len, point_stride);
            size_t num_pillars = vueron::point_decoration(
                bev_pillar, voxel_coords, voxel_num_points, pfe_input,
                (float *)points.data(), points_buf_len, point_stride);

            vueron::pfe_run(pfe_input, pfe_output);
            vueron::scatter(pfe_output, voxel_coords, num_pillars, bev_image);
            vueron::rpn_run(bev_image, rpn_outputs);
            vueron::decode_to_boxes(rpn_outputs, boxes, labels, scores);
        }
    }
    return 0;
}
