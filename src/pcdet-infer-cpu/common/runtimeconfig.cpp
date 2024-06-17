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
      nms_iou_thd(nms_iou_thd){};
