#include "config.h"
#include "model.h"
#include "npy.h"
#include "params.h"
#include "pcl.h"
#include <cstdlib>
#include <draw.h>
#include <glob.h>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <string>
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
    size_t point_stride = POINT_STRIDE;

    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    size_t num_test_files = pcd_files.size();

    for (size_t i = 0; i < num_test_files; i++) {
        std::string pcd_file = pcd_files[i];

        /*
            Read points from pcd files
        */
        std::vector<float> buffer =
            vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);
        float *points = (float *)buffer.data();
        size_t point_buf_len = buffer.size();

        /*
            Buffers for inferece
        */
        std::vector<vueron::BndBox> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<size_t> nms_labels;

        /*
            Do inference
        */
        vueron::run_model(points, point_buf_len, point_stride, nms_boxes,
                          nms_scores, nms_labels);

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
        std::cout << "vehicle(" << std::setw(3) << veh_cnt << "), pedestrian("
                  << std::setw(3) << ped_cnt << "), cyclist(" << std::setw(3)
                  << cyc_cnt << ")" << std::endl;

        auto image = drawBirdsEyeView(buffer.size() / point_stride, points,
                                      nms_boxes, nms_scores, nms_labels);
        cv::imshow("Bird's Eye View", image);
        cv::waitKey(1);
#ifdef _DEBUG
        std::string output_file_name =
            "outputs/" + std::to_string(i + 1) + ".png";
        cv::imwrite(output_file_name, image);
#endif
    }

    return 0;
}
