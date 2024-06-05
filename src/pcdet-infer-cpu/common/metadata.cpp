#include "pcdet-infer-cpu/common/metadata.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace vueron {

// ModelConfig::ModelConfig(){};

inline json ReadFile(std::string filename) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("File open failed");
  }
  json data = json::parse(file);

  file.close();
  return data;
}

class Metadata::Impl {
 public:
  json data;

  Impl() {}
};

Metadata::Metadata() : pimpl(std::make_unique<Impl>()) {}

Metadata::~Metadata() = default;

void Metadata::Setup(std::string &filename) {
  pimpl->data = ReadFile(filename);
  auto filepath = fs::path(filename);
  auto directory = filepath.parent_path();
  std::cout << "Model's metadata file: " << filename << std::endl;
  std::cout << "Model's directory: " << directory << std::endl;
  if (pimpl->data.contains("metadata_file")) {
    pimpl->data["metadata"] =
        ReadFile(directory / pimpl->data["metadata_file"]);
    std::cout << directory / pimpl->data["metadata_file"] << std::endl;
  }
  if (pimpl->data.contains("model_files")) {
    for (auto &[key, model] : pimpl->data["model_files"].items()) {
      model = (directory / model).string();
      std::cout << model << std::endl;
    }
  }
}

std::string Metadata::pfe_file() { return pimpl->data["model_files"]["pfe"]; }
std::string Metadata::rpn_file() { return pimpl->data["model_files"]["rpn"]; }

float Metadata::pillar_x_size() {
  return pimpl->data["metadata"]["voxelize"]["pillar_size"]["X"];
}
float Metadata::pillar_y_size() {
  return pimpl->data["metadata"]["voxelize"]["pillar_size"]["Y"];
}
float Metadata::pillar_z_size() {
  return pimpl->data["metadata"]["voxelize"]["pillar_size"]["Z"];
}

float Metadata::min_x_range() {
  return pimpl->data["metadata"]["voxelize"]["range"]["X"]["MIN"];
}
float Metadata::max_x_range() {
  return pimpl->data["metadata"]["voxelize"]["range"]["X"]["MAX"];
}
float Metadata::min_y_range() {
  return pimpl->data["metadata"]["voxelize"]["range"]["Y"]["MIN"];
}
float Metadata::max_y_range() {
  return pimpl->data["metadata"]["voxelize"]["range"]["Y"]["MAX"];
}
float Metadata::min_z_range() {
  return pimpl->data["metadata"]["voxelize"]["range"]["Z"]["MIN"];
}
float Metadata::max_z_range() {
  return pimpl->data["metadata"]["voxelize"]["range"]["Z"]["MAX"];
}

int Metadata::num_point_values() {
  return pimpl->data["metadata"]["voxelize"]["NUM_POINT_VALUES"];
}

bool Metadata::zero_intensity() {
  return pimpl->data["metadata"]["voxelize"]["ZERO_INTENSITY"];
}

int Metadata::max_num_points_per_pillar() {
  return pimpl->data["metadata"]["encode"]["MAX_NUM_POINTS_PER_PILLAR"];
}
int Metadata::max_voxels() {
  return pimpl->data["metadata"]["encode"]["MAX_VOXELS"];
}
int Metadata::feature_num() {
  return pimpl->data["metadata"]["encode"]["FEATURE_NUM"];
}

int Metadata::num_feature_scatter() {
  return pimpl->data["metadata"]["scatter"]["NUM_FEATURE_SCATTER"];
}
int Metadata::grid_x_size() {
  return pimpl->data["metadata"]["scatter"]["GRID_X_SIZE"];
}
int Metadata::grid_y_size() {
  return pimpl->data["metadata"]["scatter"]["GRID_Y_SIZE"];
}
int Metadata::grid_z_size() { return 1; }

int Metadata::num_classes() {
  return pimpl->data["metadata"]["post"]["CLASS_NUM"];
}
int Metadata::feature_x_size() {
  return pimpl->data["metadata"]["post"]["FEATURE_X_SIZE"];
}
int Metadata::feature_y_size() {
  return pimpl->data["metadata"]["post"]["FEATURE_Y_SIZE"];
}

std::vector<float> Metadata::iou_rectifier() {
  return pimpl->data["metadata"]["post"]["IOU_RECTIFIER"];
}

int Metadata::out_size_factor() { return (feature_x_size() / grid_x_size()); }

void Metadata::Copy() {
  modelconfig.min_x_range = Metadata::min_x_range();
  modelconfig.max_x_range = Metadata::max_x_range();
  modelconfig.min_y_range = Metadata::min_y_range();
  modelconfig.max_y_range = Metadata::max_y_range();
  modelconfig.min_z_range = Metadata::min_z_range();
  modelconfig.max_z_range = Metadata::max_z_range();

  modelconfig.pillar_x_size = Metadata::pillar_x_size();
  modelconfig.pillar_y_size = Metadata::pillar_y_size();
  modelconfig.pillar_z_size = Metadata::pillar_z_size();

  modelconfig.num_point_values = Metadata::num_point_values();
  modelconfig.zero_intensity = Metadata::zero_intensity();

  modelconfig.max_num_points_per_pillar = Metadata::max_num_points_per_pillar();
  modelconfig.max_voxels = Metadata::max_voxels();
  modelconfig.feature_num = Metadata::feature_num();

  modelconfig.num_feature_scatter = Metadata::num_feature_scatter();
  modelconfig.grid_x_size = Metadata::grid_x_size();
  modelconfig.grid_y_size = Metadata::grid_y_size();
  modelconfig.grid_z_size = Metadata::grid_z_size();

  modelconfig.num_classes = Metadata::num_classes();
  modelconfig.feature_x_size = Metadata::feature_x_size();
  modelconfig.feature_y_size = Metadata::feature_y_size();
  modelconfig.iou_rectifier = Metadata::iou_rectifier();
};

}  // namespace vueron
