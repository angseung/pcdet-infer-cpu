#include "pcdet-infer-cpu/common/metadata.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace vueron {

MetaStruct::MetaStruct(std::string& pfe_file, std::string& rpn_file,
                       float min_x_range, float max_x_range, float min_y_range,
                       float max_y_range, float min_z_range, float max_z_range,
                       float pillar_x_size, float pillar_y_size,
                       float pillar_z_size, int num_point_values,
                       bool zero_intensity, int max_num_points_per_pillar,
                       int max_voxels, int feature_num, int num_feature_scatter,
                       int grid_x_size, int grid_y_size, int grid_z_size,
                       int class_num, int feature_x_size, int feature_y_size,
                       const std::vector<float>& iou_rectifier)
    : pfe_file(pfe_file),
      rpn_file(rpn_file),
      min_x_range(min_x_range),
      max_x_range(max_x_range),
      min_y_range(min_y_range),
      max_y_range(max_y_range),
      min_z_range(min_z_range),
      max_z_range(max_z_range),
      pillar_x_size(pillar_x_size),
      pillar_y_size(pillar_y_size),
      pillar_z_size(pillar_z_size),
      num_point_values(num_point_values),
      zero_intensity(zero_intensity),
      max_num_points_per_pillar(max_num_points_per_pillar),
      max_voxels(max_voxels),
      feature_num(feature_num),
      num_feature_scatter(num_feature_scatter),
      grid_x_size(grid_x_size),
      grid_y_size(grid_y_size),
      grid_z_size(grid_z_size),
      class_num(class_num),
      feature_x_size(feature_x_size),
      feature_y_size(feature_y_size),
      iou_rectifier(iou_rectifier) {}

std::ostream& operator<<(std::ostream& os, const MetaStruct& metastruct) {
  os << "=============== Metadata ===============\n"
     << "pfe_file: " << metastruct.pfe_file << "\n"
     << "rpn_file: " << metastruct.rpn_file << "\n"
     << "min_x_range: " << metastruct.min_x_range << "\n"
     << "max_x_range: " << metastruct.max_x_range << "\n"
     << "min_y_range: " << metastruct.min_y_range << "\n"
     << "max_y_range: " << metastruct.max_y_range << "\n"
     << "min_z_range: " << metastruct.min_z_range << "\n"
     << "max_z_range: " << metastruct.max_z_range << "\n"
     << "pillar_x_size: " << metastruct.pillar_x_size << "\n"
     << "pillar_y_size: " << metastruct.pillar_y_size << "\n"
     << "pillar_z_size: " << metastruct.pillar_z_size << "\n"
     << "num_point_values: " << metastruct.num_point_values << "\n"
     << "zero_intensity: " << metastruct.zero_intensity << "\n"
     << "max_num_points_per_pillar: " << metastruct.max_num_points_per_pillar
     << "\n"
     << "max_voxels: " << metastruct.max_voxels << "\n"
     << "feature_num: " << metastruct.feature_num << "\n"
     << "num_feature_scatter: " << metastruct.num_feature_scatter << "\n"
     << "grid_x_size: " << metastruct.grid_x_size << "\n"
     << "grid_y_size: " << metastruct.grid_y_size << "\n"
     << "grid_z_size: " << metastruct.grid_z_size << "\n"
     << "class_num: " << metastruct.class_num << "\n"
     << "feature_x_size: " << metastruct.feature_x_size << "\n"
     << "feature_y_size: " << metastruct.feature_y_size << "\n"
     << "iou_rectifier: ";
  for (const auto& value : metastruct.iou_rectifier) {
    os << value << " ";
  }
  os << "\n=======================================\n";
  return os;
}

inline nlohmann::json ReadFile(const std::string& filename) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error{"File open failed\n"};
  }
  nlohmann::json data = nlohmann::json::parse(file);

  file.close();
  return data;
}

class Metadata::Impl {
 public:
  nlohmann::json data;

  Impl() = default;
};

Metadata::Metadata() : pimpl(std::make_unique<Impl>()), metastruct() {}

Metadata::~Metadata() = default;

void Metadata::Setup(const std::string& filename) {
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
    for (auto& [key, model] : pimpl->data["model_files"].items()) {
      model = (directory / model).string();
      std::cout << model << std::endl;
    }
  }

  /*
    Copy json contents into each field of Metadata::metastruct for speed issue
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
