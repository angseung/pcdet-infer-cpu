#include <glob.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <string>
#include <vector>

#include "config.h"
#include "draw/draw.h"
#include "npy.h"
#include "pcdet-infer-cpu/common/metadata.h"
#include "pcdet-infer-cpu/common/runtimeconfig.h"
#include "pcdet-infer-cpu/pcdet.h"
#include "pcl.h"

namespace fs = std::filesystem;

int main(int argc, const char **argv) {
  const std::string wd = fs::current_path().u8string();
  std::string folder_path;
  std::string metadata_path;
  if (argc < 2) {
    /*
        Case 1. Use default path for pcd and metadata
    */
    std::cout << "Usage: " << argv[0] << " <path_to_your_pcd_files_directory>"
              << std::endl;
    folder_path = PCD_PATH;
    metadata_path = wd + "/models/gcm_v4_residual/metadata.json";
    std::cout << "It will run with default pcd path: " << folder_path
              << std::endl;
    std::cout << "It will run with default metadata file: " << metadata_path
              << std::endl;
  } else if (argc == 2) {
    /*
        Case 2. Use default path for metadata
    */
    folder_path = argv[1];
    metadata_path = wd + "/models/gcm_v4_residual/metadata.json";
    std::cout << "It will run with default metadata file: " << metadata_path
              << std::endl;
  } else {
    /*
        Case 3. Use specified path for pcd and metadata
    */
    folder_path = argv[1];
    metadata_path = argv[2];
  }
  constexpr size_t point_stride = POINT_STRIDE;
  const std::vector<std::string> pcd_files =
      vueron::getPCDFileList(folder_path);
  const size_t num_test_files = pcd_files.size();

  /*
    Set Metadata Path
  */
  vueron::LoadMetadata(metadata_path);
  const auto pcdet = std::make_unique<vueron::PCDet>(PFE_FILE, RPN_FILE);

  for (size_t i = 0; i < num_test_files; i++) {
    const std::string pcd_file = pcd_files[i];

    /*
        Read points from pcd files
    */
    const std::vector<float> buffer =
        vueron::readPcdFile(pcd_file, MAX_POINT_NUM);
    const float *points = (float *)buffer.data();
    const size_t point_buf_len = buffer.size();

    /*
        Buffers for inferece
    */
    std::vector<vueron::BndBox> nms_boxes;
    std::vector<size_t> nms_labels;
    std::vector<float> nms_scores;

    /*
        Do inference
    */
    pcdet->do_infer(points, point_buf_len, point_stride, nms_boxes, nms_labels,
                    nms_scores);

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
    cv::waitKey(0);
#ifdef _DEBUG
    std::string output_file_name = "outputs/" + std::to_string(i + 1) + ".png";
    cv::imwrite(output_file_name, image);
#endif
  }

  return 0;
}
