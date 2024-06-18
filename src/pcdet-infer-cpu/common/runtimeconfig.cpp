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
