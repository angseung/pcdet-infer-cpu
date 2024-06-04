#include <glob.h>

#include "config.h"
#include "npy.h"
#include "params.h"
#include "pcdet-infer-cpu/pcdet.h"
#include "pcl.h"
#include "type.h"

int main(int argc, const char **argv) {
  const std::string folder_path = PCD_PATH;
  std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
  constexpr size_t point_stride = POINT_STRIDE;

  const auto pcdet = std::make_unique<vueron::PCDet>();

  while (1) {
    for (const auto &pcd_file : pcd_files) {
      const std::vector<float> points =
          vueron::readPcdFile(pcd_file, MAX_POINTS_NUM);
      float *point_data = (float *)points.data();
      size_t point_buf_len = points.size();
      /*
          Buffers for inference
      */
      std::vector<vueron::BndBox> nms_boxes;
      std::vector<size_t> nms_labels;
      std::vector<float> nms_scores;

      /*
          Do inference
      */
      pcdet->do_infer(point_data, point_buf_len, point_stride, nms_boxes,
                      nms_labels, nms_scores);

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
  }
  return 0;
}
