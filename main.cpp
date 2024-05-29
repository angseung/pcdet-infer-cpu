#include "config.h"
#include "model.h"
#include "npy.h"
#include "params.h"
#include "pcl.h"
#include "type.h"
#include <glob.h>
#include <iostream>

int main(int argc, const char **argv) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    size_t point_stride = NUM_POINT_VALUES;

    while (1) {
        for (const auto &pcd_file : pcd_files) {
            std::vector<float> points =
                vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);
            float *point_data = (float *)points.data();
            size_t point_buf_len = points.size();
#ifdef _DEBUG
            std::cout << file << std::endl;
            std::cout << "Points Num of " << file << ": "
                      << points.size() / sizeof(float) << std::endl;
#endif
            /*
                Buffers for inferece
            */
            std::vector<vueron::BndBox> nms_boxes;
            std::vector<float> nms_scores;
            std::vector<size_t> nms_labels;

            /*
                Do inference
            */
            vueron::run_model(point_data, point_buf_len, point_stride,
                              nms_boxes, nms_scores, nms_labels);

            /*
                Logging
            */
            auto veh_cnt = std::count_if(nms_labels.begin(), nms_labels.end(),
                                         [](int i) { return i == 1; });
            auto ped_cnt = std::count_if(nms_labels.begin(), nms_labels.end(),
                                         [](int i) { return i == 2; });
            auto cyc_cnt = std::count_if(nms_labels.begin(), nms_labels.end(),
                                         [](int i) { return i == 3; });
            std::cout << "Input file: " << pcd_file << std::endl;
            std::cout << "vehicle(" << std::setw(3) << veh_cnt
                      << "), pedestrian(" << std::setw(3) << ped_cnt
                      << "), cyclist(" << std::setw(3) << cyc_cnt << ")"
                      << std::endl;
        }
    }
    return 0;
}
