#ifndef __PCL_H__
#define __PCL_H__

#include <glob.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <cstring>
#include <filesystem>  // For std::filesystem
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace vueron {

std::vector<std::string> getPCDFileList(const std::string &folder_path) {
  std::vector<std::string> file_list;
  for (const auto &entry : fs::directory_iterator(folder_path)) {
    const std::string curr_pcd_file_name = entry.path().string();
    if (curr_pcd_file_name.find(".pcd") == std::string::npos) {
      continue;
    }
    file_list.push_back(entry.path().string());
  }
  // Sort the file list in ascending order
  std::sort(file_list.begin(), file_list.end(), std::less<std::string>());

  return file_list;
}

std::vector<std::string> getFileList(const std::string &folder_path) {
  std::vector<std::string> files;
  glob_t glob_result;

  int ret = glob(folder_path.c_str(), GLOB_TILDE, nullptr, &glob_result);

  if (ret == 0) {
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
      files.push_back(std::string(glob_result.gl_pathv[i]));
    }
  } else {
    std::cerr << "glob() failed with return code: " << ret << std::endl;
  }

  globfree(&glob_result);

  if (files.size() == 0) {
    std::cout << "There's no file in " + folder_path << std::endl;
  }
  return files;
}

std::vector<float> readPcdFile(const std::string &file_path,
                               const size_t &max_points) {
  std::vector<float> points;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZI>);

  if (pcl::io::loadPCDFile<pcl::PointXYZI>(file_path, *cloud) == -1) {
    PCL_ERROR("Couldn't read the PCD file \n");
    return points;
  }

  size_t point_count = std::min(max_points, cloud->points.size());
  points.reserve(point_count * 4);  // Reserve space for x, y, z, intensity

  for (size_t i = 0; i < point_count; ++i) {
    points.push_back(cloud->points[i].x);
    points.push_back(cloud->points[i].y);
    points.push_back(cloud->points[i].z);
    points.push_back(cloud->points[i].intensity);
  }

  return points;
}
}  // namespace vueron

#endif  // __PCL_H__
