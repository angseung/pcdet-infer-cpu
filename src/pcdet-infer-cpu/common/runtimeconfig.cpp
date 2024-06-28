#include "pcdet-infer-cpu/common/runtimeconfig.h"

RuntimeConfig::RuntimeConfig(int max_points, unsigned char shuffle_on,
                             unsigned char use_cpu, int pre_nms_max_preds,
                             int max_preds, float nms_score_thd,
                             float pre_nms_distance_thd, float nms_iou_thd)
    : max_points(max_points),
      shuffle_on(shuffle_on),
      use_cpu(use_cpu),
      pre_nms_max_preds(pre_nms_max_preds),
      max_preds(max_preds),
      nms_score_thd(nms_score_thd),
      pre_nms_distance_thd(pre_nms_distance_thd),
      nms_iou_thd(nms_iou_thd){};

std::ostream& vueron::operator<<(
    std::ostream& os,
    const vueron::RuntimeConfigSingleton& runtimeconfigsingletone) {
  os << "RuntimeConfig" << std::endl
     << "max_points: " << runtimeconfigsingletone.config.max_points << "\n"
     << "shuffle_on: "
     << static_cast<bool>(runtimeconfigsingletone.config.shuffle_on) << "\n"
     << "use_cpu: " << static_cast<bool>(runtimeconfigsingletone.config.use_cpu)
     << "\n"
     << "pre_nms_max_preds: "
     << runtimeconfigsingletone.config.pre_nms_max_preds << "\n"
     << "max_preds: " << runtimeconfigsingletone.config.max_preds << "\n"
     << "nms_score_thd: " << runtimeconfigsingletone.config.nms_score_thd
     << "\n"
     << "pre_nms_distance_thd: "
     << runtimeconfigsingletone.config.pre_nms_distance_thd << "\n"
     << "nms_iou_thd: " << runtimeconfigsingletone.config.nms_iou_thd << "\n";

  os << std::endl;

  return os;
}
