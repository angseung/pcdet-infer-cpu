#include "demo_common.h"
#include "pcdet-infer-cpu/pcdet.h"

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
    std::cout << "It will run with a default pcd path: " << pcd_path
              << std::endl;
    std::cout << "It will run with a default metadata file: " << metadata_path
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
  const auto &pcd_files = vueron::getPCDFileList(pcd_path);

  /*
    Set Metadata & RuntimeConfig
  */
  vueron::LoadMetadata(metadata_path);

  RuntimeConfig config{
      150000,  // int max_points;
      true,    // bool shuffle_on;
      true,    // bool use_cpu;
      500,     // int pre_nms_max_preds;
      83,      // int max_preds;
      0.1f,    // float nms_score_thd;
      10.0f,   // float pre_nms_distance_thd;
      0.2f,    // float nms_iou_thd;
  };

  const auto pcdet =
      std::make_unique<vueron::PCDetCPU>(PFE_FILE, RPN_FILE, &config);

  // logging version & config info
  std::cout << pcdet->getVersionInfo() << std::endl;
  std::cout << vueron::GetMetadata() << std::endl;
  std::cout << vueron::GetRuntimeConfig() << std::endl;

  for (const auto &pcd_file : pcd_files) {
    /*
        Read points from pcd files
    */
    vueron::PCDReader reader{pcd_file};
    const auto &buffer = reader.getData();
    const auto point_stride = reader.getStride();
    const auto point_buf_len = static_cast<int>(buffer.size());
    const auto *points = buffer.data();

    /*
        Buffers for inference
    */
    std::vector<Box> nms_boxes;
    std::vector<size_t> nms_labels;
    std::vector<float> nms_scores;

    /*
        Do inference
    */
    pcdet->run(points, point_buf_len, point_stride, nms_boxes, nms_labels,
               nms_scores);

    /*
        Logging
    */
    std::vector<int> indices(nms_labels.size());
    std::iota(indices.begin(), indices.end(), 0);
    const auto veh_cnt =
        std::count_if(indices.begin(), indices.end(), [&](const int j) -> bool {
          return nms_labels[j] == 0 && nms_scores[j] >= VEH_THRESHOLD;
        });
    const auto ped_cnt =
        std::count_if(indices.begin(), indices.end(), [&](const int j) -> bool {
          return nms_labels[j] == 1 && nms_scores[j] >= PED_THRESHOLD;
        });
    const auto cyc_cnt =
        std::count_if(indices.begin(), indices.end(), [&](const int j) -> bool {
          return nms_labels[j] == 2 && nms_scores[j] >= CYC_THRESHOLD;
        });
    std::cout << "Input file: " << pcd_file << std::endl;
    std::cout << "vehicle(" << std::setw(3) << veh_cnt << "), pedestrian("
              << std::setw(3) << ped_cnt << "), cyclist(" << std::setw(3)
              << cyc_cnt << ")" << std::endl;

    constexpr float scale = 12.0f;

    const int width = static_cast<int>((MAX_X_RANGE - MIN_X_RANGE) * scale);
    const int height = static_cast<int>((MAX_Y_RANGE - MIN_Y_RANGE) * scale);

    cv::Mat image(height, width, CV_8UC3, cv::Scalar{0, 0, 0});
    drawBirdsEyeView(point_buf_len, point_stride, points, nms_boxes, nms_scores,
                     nms_labels, scale, image);
    cv::imshow("Bird's Eye View", image);
    cv::waitKey(0);
#ifdef _DEBUG
    static size_t count = 0;
    std::string output_file_name =
        "./outputs/" + std::to_string(count++) + ".png";
    cv::imwrite(output_file_name, image);
#endif
  }

  std::cout << "Inference done." << std::endl;

  return 0;
}
