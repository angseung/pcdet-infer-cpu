#include "demo_c.h"

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
  const auto pcd_files = vueron::getPCDFileList(pcd_path);

  /*
    Set Metadata & Runtimeconfig
  */
  RuntimeConfig config{
      1500000,  // int max_points;
      true,     // bool shuffle_on;
      true,     // bool use_cpu;
      500,      // int pre_nms_max_preds;
      83,       // int max_preds;
      0.1f,     // float nms_score_thd;
      10.0f,    // float pre_nms_distance_thd;
      0.2f,     // float nms_iou_thd;
  };

  pcdet_initialize(metadata_path.c_str(), &config);

  // logging version & config info
  std::cout << std::string{get_pcdet_cpu_version()} << std::endl;
  std::cout << vueron::GetMetadata() << std::endl;
  std::cout << vueron::GetRuntimeConfig() << std::endl;

  for (const auto &pcd_file : pcd_files) {
    /*
        Read points from pcd files
    */
    vueron::PCDReader reader{pcd_file};
    const auto &buffer = reader.getData();
    const auto point_stride = reader.getStride();
    const int point_buf_len = static_cast<int>(buffer.size());
    const auto *points = buffer.data();

    /*
        Buffers for inferece
    */
    BndBox *preds;

    std::vector<float> nms_scores;
    std::vector<size_t> nms_labels;
    std::vector<Box> nms_boxes;
    std::vector<BndBox> nms_preds;

    /*
        Do inference
    */
    size_t n_boxes = pcdet_run(points, point_buf_len, point_stride, &preds);

    /*
        Copy predicted boxes into vector
    */
    for (size_t box_index = 0; box_index < n_boxes; box_index++) {
      BndBox pred{preds[box_index]};
      Box box{};
      nms_labels.push_back(static_cast<size_t>(pred.label));
      nms_scores.push_back(pred.score);
      box.x = pred.x;
      box.y = pred.y;
      box.z = pred.z;
      box.dx = pred.dx;
      box.dy = pred.dy;
      box.dz = pred.dz;
      box.heading = pred.heading;
      nms_boxes.push_back(box);
    }

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

    constexpr float scale = 12.0f;

    const int width = static_cast<int>((MAX_X_RANGE - MIN_X_RANGE) * scale);
    const int height = static_cast<int>((MAX_Y_RANGE - MIN_Y_RANGE) * scale);

    cv::Mat image(height, width, CV_8UC3, cv::Scalar{0, 0, 0});
    drawBirdsEyeView(point_buf_len, point_stride, points, nms_boxes, nms_scores,
                     nms_labels, scale, image);
    cv::imshow("Bird's Eye View", image);
    cv::waitKey(1);

    nms_boxes.clear();
    nms_labels.clear();
    nms_scores.clear();

#ifdef _DEBUG
    std::string output_file_name = "outputs/" + std::to_string(i + 1) + ".png";
    cv::imwrite(output_file_name, image);
#endif
  }

  pcdet_finalize();

  std::cout << "Inference done." << std::endl;

  return 0;
}
