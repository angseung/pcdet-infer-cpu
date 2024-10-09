#include "pcdet-infer-cpu/common/runtimeconfig.h"

RuntimeConfig::RuntimeConfig(const unsigned char shuffle_on,
                             const unsigned char use_cpu,
                             const float pre_nms_distance_thd)
    : shuffle_on(shuffle_on),
      use_cpu(use_cpu),
      pre_nms_distance_thd(pre_nms_distance_thd) {
  if (!use_cpu) {
    throw std::runtime_error("'use_cpu' in RuntimeConfig MUST be true.");
  }
}

std::ostream& operator<<(std::ostream& os, const RuntimeConfig& runtimeconfig) {
  os << "=============== RuntimeConfig ===============\n"
     << "shuffle_on: " << static_cast<bool>(runtimeconfig.shuffle_on) << "\n"
     << "use_cpu: " << static_cast<bool>(runtimeconfig.use_cpu) << "\n"
     << "pre_nms_distance_thd: " << runtimeconfig.pre_nms_distance_thd;

  os << "\n=============================================" << std::endl;

  return os;
}

vueron::RuntimeConfigSingleton& vueron::RuntimeConfigSingleton::Instance() {
  static RuntimeConfigSingleton config{};
  return config;
}

void vueron::RuntimeConfigSingleton::Set(const RuntimeConfig& config) {
  auto& instance = Instance();
  instance.config = config;
}
