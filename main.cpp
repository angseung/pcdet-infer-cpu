#include "main.h"

int main(int argc, const char **argv) {
  const std::string wd = fs::current_path().u8string();
  std::string pcd_path;
  std::string metadata_path;

  if (argc < 2) {
    /*
        Case 1. Use default path for pcd and metadata
    */
    std::cout << "Usage: " << argv[0] << " <path_to_your_pcd_files_directory>"
              << " <path_to_your_metadata_files_directory>" << std::endl;
    pcd_path = "./pcd/cepton";
    metadata_path = wd + "/models/gcm_v4_residual/metadata.json";
    std::cout << "It will run with default pcd path: " << pcd_path << std::endl;
    std::cout << "It will run with default metadata file: " << metadata_path
              << std::endl;
  } else if (argc == 2) {
    /*
        Case 2. Use default path for metadata
    */
    pcd_path = argv[1];
    metadata_path = wd + "/models/gcm_v4_residual/metadata.json";
    std::cout << "It will run with default metadata file: " << metadata_path
              << std::endl;
  } else {
    /*
        Case 3. Use specified path for pcd and metadata
    */
    pcd_path = argv[1];
    metadata_path = argv[2];
  }
  const std::vector<std::string> pcd_files = vueron::getPCDFileList(pcd_path);

  /*
    Set Metadata & Runtimeconfig
  */
  vueron::LoadMetadata(metadata_path);

  RuntimeConfig config{
      1000000,  // int max_points;
      false,    // bool shuffle_on;
      true,     // bool use_cpu;
      500,      // int pre_nms_max_preds;
      83,       // int max_preds;
      0.1f,     // float nms_score_thd;
      10.0f,    // float pre_nms_distance_thd;
      0.2f,     // float nms_iou_thd;
  };

  std::cout << vueron::GetMetadata() << std::endl;
  std::cout << vueron::GetRuntimeConfig() << std::endl;

  /*
    Init PCDetCPU with metadata & runtimeconfig
  */
  const auto pcdet =
      std::make_unique<vueron::PCDetCPU>(PFE_FILE, RPN_FILE, &config);
  std::cout << pcdet->getVersionInfo() << std::endl;

  for (const auto &pcd_file : pcd_files) {
    /*
      Read point data from pcd files
    */
    vueron::PCDReader reader(pcd_file);
    const std::vector<float> points = reader.getData();
    const size_t point_stride = reader.getStride();
    const size_t point_buf_len = points.size();
    const float *point_data = points.data();

    /*
        Buffers for inference
    */
    std::vector<vueron::BndBox> nms_boxes;
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
  }

  return 0;
}
