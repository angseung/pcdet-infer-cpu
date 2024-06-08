#include "pcdet-infer-cpu/common/metadata.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace vueron {

inline json ReadFile(const std::string &filename) {
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

  Impl() = default;
};

Metadata::Metadata() : pimpl(std::make_unique<Impl>()) {}

Metadata::~Metadata() = default;

void Metadata::Setup(const std::string &filename) const {
  pimpl->data = ReadFile(filename);
  const auto filepath = fs::path(filename);
  const auto directory = filepath.parent_path();
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
  metastruct.pfe_name = pimpl->data["model_files"]["pfe"];
  metastruct.rpn_file = pimpl->data["model_files"]["rpn"];

  metastruct.min_x_range =
      pimpl->data["metadata"]["voxelize"]["range"]["X"]["MIN"];
  metastruct.max_x_range =
      pimpl->data["metadata"]["voxelize"]["range"]["X"]["MAX"];
  metastruct.min_y_range =
      pimpl->data["metadata"]["voxelize"]["range"]["Y"]["MIN"];
  metastruct.max_y_range =
      pimpl->data["metadata"]["voxelize"]["range"]["Y"]["MAX"];
  metastruct.min_z_range =
      pimpl->data["metadata"]["voxelize"]["range"]["Z"]["MIN"];
  metastruct.max_z_range =
      pimpl->data["metadata"]["voxelize"]["range"]["Z"]["MAX"];

  metastruct.pillar_x_size =
      pimpl->data["metadata"]["voxelize"]["pillar_size"]["X"];
  metastruct.pillar_y_size =
      pimpl->data["metadata"]["voxelize"]["pillar_size"]["Y"];
  metastruct.pillar_z_size =
      pimpl->data["metadata"]["voxelize"]["pillar_size"]["Z"];

  metastruct.num_point_values =
      pimpl->data["metadata"]["voxelize"]["NUM_POINT_VALUES"];
  metastruct.zero_intensity =
      pimpl->data["metadata"]["voxelize"]["ZERO_INTENSITY"];

  metastruct.max_num_points_per_pillar =
      pimpl->data["metadata"]["encode"]["MAX_NUM_POINTS_PER_PILLAR"];
  metastruct.max_voxels = pimpl->data["metadata"]["encode"]["MAX_VOXELS"];
  metastruct.feature_num = pimpl->data["metadata"]["encode"]["FEATURE_NUM"];

  metastruct.num_feature_scatter =
      pimpl->data["metadata"]["scatter"]["NUM_FEATURE_SCATTER"];
  metastruct.grid_x_size = pimpl->data["metadata"]["scatter"]["GRID_X_SIZE"];
  metastruct.grid_y_size = pimpl->data["metadata"]["scatter"]["GRID_Y_SIZE"];
  metastruct.grid_z_size = 1;

  metastruct.num_classes = pimpl->data["metadata"]["post"]["CLASS_NUM"];
  metastruct.feature_x_size = pimpl->data["metadata"]["post"]["FEATURE_X_SIZE"];
  metastruct.feature_y_size = pimpl->data["metadata"]["post"]["FEATURE_Y_SIZE"];
  metastruct.iou_rectifier = static_cast<std::vector<float>>(
      pimpl->data["metadata"]["post"]["IOU_RECTIFIER"]);
}

MetaStruct Metadata::metastruct = {
    "",   "",    0.0f, 0.0f,
    0.0f, 0.0f,  0.0f, 0.0f,

    0.0f, 0.0f,  0.0f,

    4,    false,

    20,   25000, 10,

    64,   224,   328,  1,

    3,    112,   164,  {0.0f, 0.0f, 0.0f},

};
std::string Metadata::pfe_file() const { return metastruct.pfe_name; }
std::string Metadata::rpn_file() const { return metastruct.rpn_file; }

float Metadata::min_x_range() const { return metastruct.min_x_range; }
float Metadata::max_x_range() const { return metastruct.max_x_range; }
float Metadata::min_y_range() const { return metastruct.min_y_range; }
float Metadata::max_y_range() const { return metastruct.max_y_range; }
float Metadata::min_z_range() const { return metastruct.min_z_range; }
float Metadata::max_z_range() const { return metastruct.max_z_range; }

float Metadata::pillar_x_size() const { return metastruct.pillar_x_size; }
float Metadata::pillar_y_size() const { return metastruct.pillar_y_size; }
float Metadata::pillar_z_size() const { return metastruct.pillar_z_size; }

int Metadata::num_point_values() const { return metastruct.num_point_values; }
bool Metadata::zero_intensity() const { return metastruct.zero_intensity; }

int Metadata::max_num_points_per_pillar() const {
  return metastruct.max_num_points_per_pillar;
}
int Metadata::max_voxels() const { return metastruct.max_voxels; }
int Metadata::feature_num() const { return metastruct.feature_num; }

int Metadata::num_feature_scatter() const {
  return metastruct.num_feature_scatter;
}
int Metadata::grid_x_size() const { return metastruct.grid_x_size; }
int Metadata::grid_y_size() const { return metastruct.grid_y_size; }
int Metadata::grid_z_size() const { return metastruct.grid_z_size; }

int Metadata::num_classes() const { return metastruct.num_classes; }
int Metadata::feature_x_size() const { return metastruct.feature_x_size; }
int Metadata::feature_y_size() const { return metastruct.feature_y_size; }
std::vector<float> Metadata::iou_rectifier() const {
  return metastruct.iou_rectifier;
}

}  // namespace vueron
