#ifndef __RUNTIMECONFIG_H__
#define __RUNTIMECONFIG_H__

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

struct RuntimeConfig {
  int max_points;
  unsigned char shuffle_on;
  unsigned char use_cpu;
  int pre_nms_max_preds;
  int max_preds;
  float nms_score_thd;
  float pre_nms_distance_thd;
  float nms_iou_thd;

  RuntimeConfig(int max_points = 1000000, unsigned char shuffle_on = false,
                unsigned char use_cpu = true, int pre_nms_max_preds = 500,
                int max_preds = 83, float nms_score_thd = 0.1f,
                float pre_nms_distance_thd = 10.0f, float nms_iou_thd = 0.2f);
  RuntimeConfig();
  ~RuntimeConfig();
};

#ifdef __cplusplus
namespace vueron {

struct RuntimeConfigSingleton {
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

  static RuntimeConfigSingleton& Instance() {
    static RuntimeConfigSingleton config;
    return config;
  }
  static void Set(const RuntimeConfig& config) {
    auto& instance = Instance();
    instance.config = config;
  }
};

inline auto& GetRuntimeConfig() {
  return RuntimeConfigSingleton::Instance().config;
}
inline void SetRuntimeConfig(const RuntimeConfig& config) {
  RuntimeConfigSingleton::Set(config);
}

}  // namespace vueron

#define MAX_POINT_NUM vueron::GetRuntimeConfig().max_points
#define SHUFFLE_ON vueron::GetRuntimeConfig().shuffle_on
#define USE_CPU vueron::GetRuntimeConfig().use_cpu
#define NMS_PRE_MAXSIZE vueron::GetRuntimeConfig().pre_nms_max_preds
#define MAX_OBJ_PER_SAMPLE vueron::GetRuntimeConfig().max_preds
#define SCORE_THRESH vueron::GetRuntimeConfig().nms_score_thd
#define PRE_NMS_DISTANCE_THD vueron::GetRuntimeConfig().pre_nms_distance_thd
#define NMS_THRESH vueron::GetRuntimeConfig().nms_iou_thd
#define INTENSITY_NORMALIZE_DIV 255
#define CONF_THRESH 0.4f
#define RANDOM_SEED 123
#define POINT_STRIDE 4

#endif

#endif  // __RUNTIMECONFIG_H__
