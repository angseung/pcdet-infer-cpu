#include "pcdet-infer-cpu/common/runtimeconfig.h"

RuntimeConfig::RuntimeConfig(const int max_points,
                             const unsigned char shuffle_on,
                             const unsigned char use_cpu,
                             const int pre_nms_max_preds, const int max_preds,
                             const float nms_score_thd,
                             const float pre_nms_distance_thd,
                             const float nms_iou_thd)
    : max_points(max_points),
      shuffle_on(shuffle_on),
      use_cpu(use_cpu),
      pre_nms_max_preds(pre_nms_max_preds),
      max_preds(max_preds),
      nms_score_thd(nms_score_thd),
      pre_nms_distance_thd(pre_nms_distance_thd),
      nms_iou_thd(nms_iou_thd) {}

std::ostream& operator<<(std::ostream& os, const RuntimeConfig& runtimeconfig) {
  os << "=============== RuntimeConfig ===============\n"
     << "max_points: " << runtimeconfig.max_points << "\n"
     << "shuffle_on: " << static_cast<bool>(runtimeconfig.shuffle_on) << "\n"
     << "use_cpu: " << static_cast<bool>(runtimeconfig.use_cpu) << "\n"
     << "pre_nms_max_preds: " << runtimeconfig.pre_nms_max_preds << "\n"
     << "max_preds: " << runtimeconfig.max_preds << "\n"
     << "nms_score_thd: " << runtimeconfig.nms_score_thd << "\n"
     << "pre_nms_distance_thd: " << runtimeconfig.pre_nms_distance_thd << "\n"
     << "nms_iou_thd: " << runtimeconfig.nms_iou_thd;

  os << "\n=============================================" << std::endl;

  return os;
}
