#ifndef __RUNTIMECONFIG_H__
#define __RUNTIMECONFIG_H__

#ifdef __cplusplus
#include <cstddef>
#include <iostream>
#else
#include <stddef.h>
#endif

struct RuntimeConfig {
  unsigned char shuffle_on;
  unsigned char use_cpu;  // Reserved
  float pre_nms_distance_thd;

#ifdef __cplusplus
  RuntimeConfig() = delete;
  explicit RuntimeConfig(unsigned char shuffle_on, unsigned char use_cpu,
                         float pre_nms_distance_thd);
  ~RuntimeConfig() = default;
#endif
};

#ifdef __cplusplus
std::ostream& operator<<(std::ostream& os, const RuntimeConfig& runtimeconfig);

namespace vueron {

struct RuntimeConfigSingleton {
  RuntimeConfig config{
      false,  // bool shuffle_on;
      true,   // bool use_cpu;
      10.0f,  // float pre_nms_distance_thd;
  };
  RuntimeConfigSingleton(const RuntimeConfigSingleton& copy) = delete;
  RuntimeConfigSingleton& operator=(const RuntimeConfigSingleton& copy) =
      delete;
  RuntimeConfigSingleton(const RuntimeConfigSingleton&& rhs) = delete;
  RuntimeConfigSingleton& operator=(const RuntimeConfigSingleton&& rhs) =
      delete;
  static RuntimeConfigSingleton& Instance();
  static void Set(const RuntimeConfig& config);
};

inline RuntimeConfig& GetRuntimeConfig() {
  return RuntimeConfigSingleton::Instance().config;
}
inline void SetRuntimeConfig(const RuntimeConfig& config) {
  RuntimeConfigSingleton::Set(config);
}

}  // namespace vueron

#define SHUFFLE_ON vueron::GetRuntimeConfig().shuffle_on
#define USE_CPU vueron::GetRuntimeConfig().use_cpu
#define PRE_NMS_DISTANCE_THD vueron::GetRuntimeConfig().pre_nms_distance_thd
#define RANDOM_SEED 123

#endif  //__cplusplus

#endif  // __RUNTIMECONFIG_H__
