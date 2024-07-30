#include "pcl.h"

#include <glob.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>  // std::exit and EXIT_FAILURE
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace fs = std::filesystem;

vueron::PCDReader::PCDReader(const std::string &filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << filePath << std::endl;
  }

  std::string line;
  std::map<std::string, int> fieldOffsets;
  int pointSize = 0;
  int numPoints = 0;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string token;
    iss >> token;

    if (token == "FIELDS") {
      std::string field;
      stride = 0;
      while (iss >> field) {
        fieldOffsets[field] = static_cast<int>(stride * sizeof(float));
        stride++;
      }
      pointSize = static_cast<int>(stride * sizeof(float));
    } else if (token == "DATA") {
      std::string dataType;
      iss >> dataType;
      if (dataType != "binary") {
        std::cerr << "This reader only supports binary data." << std::endl;
      }
      break;
    } else if (token == "POINTS") {
      std::string point_num_string =
          line.replace(line.begin(), line.begin() + 6, "");
      numPoints = std::stoi(point_num_string);
    }
  }

  std::vector<char> buffer(pointSize);
  while (file.read(buffer.data(), pointSize)) {
    for (int i = 0; i < stride; i++) {
      float value;
      memcpy(&value, buffer.data() + i * sizeof(float), sizeof(float));

      // check parsed point value is NaN or not.
      if (std::isnan(value)) {
        std::cerr << "ERROR: NaN value encountered in the file." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      data.push_back(value);
    }
  }

  file.close();

  // check num_points
  if (numPoints != data.size() * sizeof(float) / pointSize) {
    std::cerr << "Parsed point_num is difference with num_points in pcd header"
              << std::endl;
  }
}

const std::vector<float> &vueron::PCDReader::getData() const { return data; }

int vueron::PCDReader::getStride() const { return stride; }

std::vector<std::string> vueron::getPCDFileList(
    const std::string &folder_path) {
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

std::vector<std::string> vueron::getFileList(const std::string &folder_path) {
  std::vector<std::string> files;
  glob_t glob_result;

  int ret = glob(folder_path.c_str(), GLOB_TILDE, nullptr, &glob_result);

  if (ret == 0) {
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
      files.emplace_back(glob_result.gl_pathv[i]);
    }
  } else {
    std::cerr << "glob() failed with return code: " << ret << std::endl;
  }

  globfree(&glob_result);

  if (files.empty()) {
    std::cout << "There's no file in " + folder_path << std::endl;
  }
  return files;
}
