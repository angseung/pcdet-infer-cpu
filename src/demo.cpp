#include "config.h"
#include "params.h"
#include "pcl.h"
#include "post.h"
#include "pre.h"
#include "rpn.h"
#include <cstdlib>
#include <draw/draw.h>
#include <glob.h>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>

int main(int argc, const char **argv) {
    std::string folder_path;
    if (argc < 2) {
        std::cout << "Usage: " << argv[0]
                  << " <path_to_your_pcd_files_directory>" << std::endl;
        folder_path = PCD_PATH;
        std::cout << "It will run with default pcd path: " << folder_path
                  << std::endl;
    } else {
        folder_path = argv[1];
    }
    // bool do_memory_check = false;

    size_t point_stride = NUM_POINT_VALUES;

    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    std::vector<float> buffer;

    for (const auto &file : pcd_files) {
        std::vector<float> buffer = vueron::readPcdFile(file, MAX_POINTS_NUM);
        float *points_data = (float *)buffer.data();
        size_t points_buf_len = buffer.size();

        /*
            Buffers for inferece
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
                                     0.0f);          // input of rpn_run()
        std::vector<std::vector<float>> rpn_outputs; // output of rpn_run()
        std::vector<vueron::BndBox> boxes(
            MAX_BOX_NUM_BEFORE_NMS);                        // boxes before NMS
        std::vector<size_t> labels(MAX_BOX_NUM_BEFORE_NMS); // labels before NMS
        std::vector<float> scores(MAX_BOX_NUM_BEFORE_NMS);  // scores before NMS

        /*
            Do inference
        */
        vueron::voxelization(bev_pillar, points_data, points_buf_len,
                             point_stride);
        size_t num_pillars = vueron::point_decoration(
            bev_pillar, voxel_coords, voxel_num_points, pfe_input, points_data,
            points_buf_len, point_stride);

        vueron::pfe_run(pfe_input, pfe_output);
        vueron::scatter(pfe_output, voxel_coords, num_pillars, bev_image);
        vueron::rpn_run(bev_image, rpn_outputs);
        vueron::decode_to_boxes(rpn_outputs, boxes, labels, scores);

        /*
            Logging
        */
        auto veh_cnt = std::count_if(labels.begin(), labels.end(),
                                     [](int i) { return i == 1; });
        auto ped_cnt = std::count_if(labels.begin(), labels.end(),
                                     [](int i) { return i == 2; });
        auto cyc_cnt = std::count_if(labels.begin(), labels.end(),
                                     [](int i) { return i == 3; });
        std::cout << "Input file: " << file << std::endl;
        std::cout << "vehicle(" << std::setw(3) << veh_cnt << "), pedestrian("
                  << std::setw(3) << ped_cnt << "), cyclist(" << std::setw(3)
                  << cyc_cnt << ")" << std::endl;

        auto image = drawBirdsEyeView(buffer.size() / point_stride, points_data,
                                      boxes, scores, labels);
        cv::imshow("Bird's Eye View", image);
        cv::waitKey(1);
    }

    return 0;
}
