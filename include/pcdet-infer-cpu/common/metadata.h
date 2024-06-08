#ifndef __METADATA_H__
#define __METADATA_H__

#include <memory>
#include <string>
#include <vector>

namespace vueron {
struct MetaStruct {
  std::string pfe_file;
  std::string rpn_file;

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

  int max_num_points_per_pillar;
  int max_voxels;
  int feature_num;

  int num_feature_scatter;
  int grid_x_size;
  int grid_y_size;
  int grid_z_size;

  int class_num;
  int feature_x_size;
  int feature_y_size;
  std::vector<float> iou_rectifier;
};

class Metadata {
 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
  void Setup(const std::string& filename);

 public:
  Metadata();
  ~Metadata();
  MetaStruct metastruct{
      "",   "",    0.0f, 0.0f,
      0.0f, 0.0f,  0.0f, 0.0f,

      0.0f, 0.0f,  0.0f,

      4,    false,

      20,   25000, 10,

      64,   224,   328,  1,

      3,    112,   164,  {0.0f, 0.0f, 0.0f},

  };
  static Metadata& Instance() {
    static Metadata metadata;
    return metadata;
  }
  static void Load(const std::string& filename) {
    auto& instance = Instance();
    instance.Setup(filename);
  }
};

inline auto& GetMetadata() { return Metadata::Instance().metastruct; }
inline void LoadMetadata(const std::string& filename) {
  Metadata::Load(filename);
}

}  // namespace vueron

#define PFE_FILE vueron::GetMetadata().pfe_file
#define RPN_FILE vueron::GetMetadata().rpn_file

#define MIN_X_RANGE vueron::GetMetadata().min_x_range
#define MAX_X_RANGE vueron::GetMetadata().max_x_range
#define MIN_Y_RANGE vueron::GetMetadata().min_y_range
#define MAX_Y_RANGE vueron::GetMetadata().max_y_range
#define MIN_Z_RANGE vueron::GetMetadata().min_z_range
#define MAX_Z_RANGE vueron::GetMetadata().max_z_range

#define PILLAR_X_SIZE vueron::GetMetadata().pillar_x_size
#define PILLAR_Y_SIZE vueron::GetMetadata().pillar_y_size
#define PILLAR_Z_SIZE vueron::GetMetadata().pillar_z_size

#define NUM_POINT_VALUES vueron::GetMetadata().num_point_values
#define ZERO_INTENSITY vueron::GetMetadata().zero_intensity

#define MAX_NUM_POINTS_PER_PILLAR \
  vueron::GetMetadata().max_num_points_per_pillar
#define MAX_VOXELS vueron::GetMetadata().max_voxels
#define FEATURE_NUM vueron::GetMetadata().feature_num

#define NUM_FEATURE_SCATTER vueron::GetMetadata().num_feature_scatter
#define GRID_X_SIZE vueron::GetMetadata().grid_x_size
#define GRID_Y_SIZE vueron::GetMetadata().grid_y_size
#define GRID_Z_SIZE vueron::GetMetadata().grid_z_size

#define CLASS_NUM vueron::GetMetadata().class_num
#define FEATURE_X_SIZE vueron::GetMetadata().feature_x_size
#define FEATURE_Y_SIZE vueron::GetMetadata().feature_y_size
#define IOU_RECTIFIER vueron::GetMetadata().iou_rectifier

#define POINT_STRIDE 4
#define INTENSITY_NORMALIZE_DIV 255

#endif  // __METADATA_H__
