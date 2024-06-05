#ifndef __METADATA_H__
#define __METADATA_H__

#include <memory>
#include <string>
#include <vector>

namespace vueron {

struct ModelConfig {
  /*
      Params for Preprocessing
  */
  float min_x_range;
  float max_x_range;
  float min_y_range;
  float max_y_range;
  float min_z_range;
  float max_z_range;

  float pillar_x_size;
  float pillar_y_size;
  float pillar_z_size;

  int num_point_values;
  bool zero_intensity;

  // encode
  int max_num_points_per_pillar;
  int max_voxels;
  int feature_num;

  // scatter
  int num_feature_scatter;
  int grid_x_size;
  int grid_y_size;
  int grid_z_size;

  /*
      Params for Postprocessing
  */
  // post
  int num_classes;
  int feature_x_size;
  int feature_y_size;
  std::vector<float> iou_rectifier;
};

class Metadata {
 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
  void Setup(std::string& filename);
  void Copy();

 public:
  Metadata();
  ~Metadata();
  static Metadata& Instance() {
    static Metadata metadata;
    return metadata;
  }
  static void Load(std::string& filename) {
    auto& instance = Instance();
    instance.Setup(filename);
  }
  static ModelConfig modelconfig;

  std::string pfe_file();
  std::string rpn_file();

  float pillar_x_size();
  float pillar_y_size();
  float pillar_z_size();

  float min_x_range();
  float max_x_range();
  float min_y_range();
  float max_y_range();
  float min_z_range();
  float max_z_range();

  int num_point_values();
  bool zero_intensity();

  int max_num_points_per_pillar();
  int max_voxels();
  int feature_num();

  int num_feature_scatter();
  int grid_x_size();
  int grid_y_size();
  int grid_z_size();

  int num_classes();
  int feature_x_size();
  int feature_y_size();
  std::vector<float> iou_rectifier();

  int out_size_factor();
};

inline auto& GetMetadata() { return Metadata::Instance(); }
inline void LoadMetadata(std::string& filename) { Metadata::Load(filename); }

}  // namespace vueron

#define PFE_FILE vueron::GetMetadata().pfe_file()
#define RPN_FILE vueron::GetMetadata().rpn_file()

#define MIN_X_RANGE vueron::GetMetadata().min_x_range()
#define MAX_X_RANGE vueron::GetMetadata().max_x_range()
#define MIN_Y_RANGE vueron::GetMetadata().min_y_range()
#define MAX_Y_RANGE vueron::GetMetadata().max_y_range()
#define MIN_Z_RANGE vueron::GetMetadata().min_z_range()
#define MAX_Z_RANGE vueron::GetMetadata().max_z_range()

#define PILLAR_X_SIZE vueron::GetMetadata().pillar_x_size()
#define PILLAR_Y_SIZE vueron::GetMetadata().pillar_y_size()
#define PILLAR_Z_SIZE vueron::GetMetadata().pillar_z_size()

#define NUM_POINT_VALUES vueron::GetMetadata().num_point_values()
#define ZERO_INTENSITY vueron::GetMetadata().zero_intensity()

#define MAX_NUM_POINTS_PER_PILLAR \
  vueron::GetMetadata().max_num_points_per_pillar()
#define MAX_VOXELS vueron::GetMetadata().max_voxels()
#define FEATURE_NUM vueron::GetMetadata().feature_num()

#define NUM_FEATURE_SCATTER vueron::GetMetadata().num_feature_scatter()
#define GRID_X_SIZE vueron::GetMetadata().grid_x_size()
#define GRID_Y_SIZE vueron::GetMetadata().grid_y_size()
#define GRID_Z_SIZE vueron::GetMetadata().grid_z_size()

#define NUM_CLASSES vueron::GetMetadata().num_classes()
#define FEATURE_X_SIZE vueron::GetMetadata().feature_x_size()
#define FEATURE_Y_SIZE vueron::GetMetadata().feature_y_size()
#define IOU_RECTIFIER vueron::GetMetadata().iou_rectifier()

#define CLASS_NUM NUM_CLASSES
#define STRIDE_FOR_PILLARS MAX_NUM_POINTS_PER_PILLAR
#define MAX_NUM_PILLARS MAX_VOXELS
#define PILLARPOINTS_BEV (MAX_NUM_POINTS_PER_PILLAR * MAX_NUM_PILLARS)
#define OUT_SIZE_FACTOR (GRID_X_SIZE / FEATURE_X_SIZE)

#define CONF_THRESH 0.4f
#define INTENSITY_NORMALIZE_DIV 255

#endif  // __METADATA_H__
