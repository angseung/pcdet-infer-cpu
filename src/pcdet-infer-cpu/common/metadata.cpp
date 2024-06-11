#include "pcdet-infer-cpu/common/metadata.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <json.hpp>

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

void Metadata::Setup(const std::string &filename) {
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

  /*
    Copy Json contents into each field of Metadata::metastruct for speed issue
  */
  metastruct.pfe_file = pimpl->data["model_files"]["pfe"];
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

  metastruct.class_num = pimpl->data["metadata"]["post"]["CLASS_NUM"];
  metastruct.feature_x_size = pimpl->data["metadata"]["post"]["FEATURE_X_SIZE"];
  metastruct.feature_y_size = pimpl->data["metadata"]["post"]["FEATURE_Y_SIZE"];
  metastruct.iou_rectifier = static_cast<std::vector<float>>(
      pimpl->data["metadata"]["post"]["IOU_RECTIFIER"]);
}

}  // namespace vueron
