#include "config.h"
#include "model.h"
#include "npy.h"
#include "params.h"
#include "pcl.h"
#include <cstdlib>
#include <draw/draw.h>
#include <glob.h>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>

// #define FROM_SNAPSHOT

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
    size_t point_stride = NUM_POINT_VALUES;

    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
#ifdef FROM_SNAPSHOT
    std::string snapshot_folder_path = SNAPSHOT_PATH;
    std::vector<std::string> snapshot_files =
        vueron::getFileList(snapshot_folder_path);
#endif
    size_t num_test_files = pcd_files.size();

    for (size_t i = 0; i < num_test_files; i++) {
        std::string pcd_file = pcd_files[i];
#ifdef FROM_SNAPSHOT
        std::string snapshot_dir = snapshot_files[i];
#endif

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
        std::vector<vueron::BndBox> boxes; // boxes before NMS
        std::vector<size_t> labels;        // labels before NMS
        std::vector<float> scores;         // scores before NMS

        boxes.reserve(MAX_BOX_NUM_BEFORE_NMS);
        labels.reserve(MAX_BOX_NUM_BEFORE_NMS);
        scores.reserve(MAX_BOX_NUM_BEFORE_NMS);

        /*
            Do inference
        */
        vueron::run_model(points, point_buf_len, point_stride, boxes, labels,
                          scores);

#ifdef FROM_SNAPSHOT
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

        std::vector<vueron::BndBox> s_boxes(scores_snapshot.size());
        std::vector<float> s_scores(scores_snapshot.size());
        std::vector<size_t> s_labels(labels_snapshot.size());

        assert(boxes_snapshot.size() == 7 * labels_snapshot.size());
        assert(scores_snapshot.size() == labels_snapshot.size());

        for (size_t j = 0; j < labels_snapshot.size(); j++) {
            s_scores[j] = scores_snapshot[j];
            s_labels[j] = (size_t)labels_snapshot[j];
            s_boxes[j].x = boxes_snapshot[7 * j];
            s_boxes[j].y = boxes_snapshot[7 * j + 1];
            s_boxes[j].z = boxes_snapshot[7 * j + 2];
            s_boxes[j].dx = boxes_snapshot[7 * j + 3];
            s_boxes[j].dy = boxes_snapshot[7 * j + 4];
            s_boxes[j].dz = boxes_snapshot[7 * j + 5];
            s_boxes[j].heading = boxes_snapshot[7 * j + 6];
        }
#endif
        /*
            Logging
        */
        auto veh_cnt = std::count_if(labels.begin(), labels.end(),
                                     [](int i) { return i == 1; });
        auto ped_cnt = std::count_if(labels.begin(), labels.end(),
                                     [](int i) { return i == 2; });
        auto cyc_cnt = std::count_if(labels.begin(), labels.end(),
                                     [](int i) { return i == 3; });
        std::cout << "Input file: " << pcd_file << std::endl;
        std::cout << "vehicle(" << std::setw(3) << veh_cnt << "), pedestrian("
                  << std::setw(3) << ped_cnt << "), cyclist(" << std::setw(3)
                  << cyc_cnt << ")" << std::endl;

#ifdef FROM_SNAPSHOT
        auto image = drawBirdsEyeView(buffer.size() / point_stride, points,
                                      s_boxes, s_scores, s_labels);
#else
        auto image = drawBirdsEyeView(buffer.size() / point_stride, points,
                                      boxes, scores, labels);
#endif
        cv::imshow("Bird's Eye View", image);
        cv::waitKey(1);
    }

    return 0;
}
