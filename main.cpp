#include "main.h"

namespace fs = std::filesystem;

int main(int argc, const char **argv) {
  const std::string wd = fs::current_path().u8string();
  std::string pcd_path{};
  std::string metadata_path{};

  if (argc < 2) {
    /*
        Case 1. Use default path for pcd and metadata
    */
    std::cout << "Usage: " << argv[0] << " <path_to_your_pcd_files_directory>"
              << " <path_to_your_metadata_files_directory>" << std::endl;
    pcd_path = "./pcd/cepton";
    metadata_path = wd + "/models/gcm_v4_residual/metadata.json";
    std::cout << "Run with a default pcd path: " << pcd_path << std::endl;
    std::cout << "Run with a default metadata file: " << metadata_path
              << std::endl;
  } else if (argc == 2) {
    /*
        Case 2. Use default path for metadata
    */
    pcd_path = argv[1];
    metadata_path = wd + "/models/gcm_v4_residual/metadata.json";
    std::cout << "It will run with a default metadata file: " << metadata_path
              << std::endl;
  } else {
    /*
        Case 3. Use specified path for pcd and metadata
    */
    pcd_path = argv[1];
    metadata_path = argv[2];
  }
  const auto pcd_files = vueron::getPCDFileList(pcd_path);

  /*
    Set Metadata & RuntimeConfig
  */
  vueron::LoadMetadata(metadata_path);

  RuntimeConfig config{
      false,  // bool shuffle_on;
      true,   // bool use_cpu;
      10.0f,  // float pre_nms_distance_thd;
  };

  /*
    Init PCDetCPU with metadata & runtimeconfig
  */
  const auto pcdet =
      std::make_unique<vueron::PCDetCPU>(PFE_FILE, RPN_FILE, &config);

  // logging Metadata & RuntimeConfig
  std::cout << vueron::GetMetadata() << std::endl;
  std::cout << vueron::GetRuntimeConfig() << std::endl;

  // logging version info
  std::cout << pcdet->version_info << std::endl;

  for (const auto &pcd_file : pcd_files) {
    /*
      Read point data from pcd files
    */
    vueron::PCDReader reader{pcd_file};
    const auto &points = reader.getXYZI();
    constexpr int point_stride = 4;
    const auto point_buf_len = points.size();
    const auto *point_data = points.data();

    /*
        Buffers for inference
    */
    std::vector<Box> nms_boxes;
    std::vector<size_t> nms_labels;
    std::vector<float> nms_scores;

    /*
        Do inference
    */
    pcdet->run(point_data, point_buf_len, point_stride, nms_boxes, nms_labels,
               nms_scores);

    /*
        Logging
    */
    const auto veh_cnt =
        std::count_if(nms_labels.begin(), nms_labels.end(),
                      [](const int j) -> bool { return j == 0; });
    const auto ped_cnt =
        std::count_if(nms_labels.begin(), nms_labels.end(),
                      [](const int j) -> bool { return j == 1; });
    const auto cyc_cnt =
        std::count_if(nms_labels.begin(), nms_labels.end(),
                      [](const int j) -> bool { return j == 2; });
    std::cout << "Input file: " << pcd_file << std::endl;
    std::cout << "vehicle(" << std::setw(3) << veh_cnt << "), pedestrian("
              << std::setw(3) << ped_cnt << "), cyclist(" << std::setw(3)
              << cyc_cnt << ")" << std::endl;
  }

  std::cout << "Inference done." << std::endl;

  return 0;
}
