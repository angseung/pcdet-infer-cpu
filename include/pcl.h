#ifndef __PCL_H__
#define __PCL_H__

#include <glob.h>

#include <filesystem>  // For std::filesystem
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace vueron {

class PCDReader {
 private:
  std::vector<float> data;
  int stride = 0;

 public:
  PCDReader(const std::string &filePath, int expected_point_num = 0);

  const std::vector<float> &getData() const { return data; }

  int getStride() const { return stride; }
};

PCDReader::PCDReader(const std::string &filePath, int expected_point_num) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << filePath << std::endl;
  }

  std::string line;
  std::map<std::string, int> fieldOffsets;
  int pointSize = 0;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string token;
    iss >> token;

    if (token == "FIELDS") {
      std::string field;
      stride = 0;
      while (iss >> field) {
        fieldOffsets[field] = stride * sizeof(float);
        stride++;
      }
      pointSize = stride * sizeof(float);
    } else if (token == "DATA") {
      std::string dataType;
      iss >> dataType;
      if (dataType != "binary") {
        std::cerr << "This reader only supports binary data." << std::endl;
      }
      break;
    }
  }

  std::vector<char> buffer(pointSize);
  while (file.read(buffer.data(), pointSize)) {
    for (int i = 0; i < stride; i++) {
      if (expected_point_num != 0 && i == expected_point_num) {
        break;
      }
      float value;
      memcpy(&value, buffer.data() + i * sizeof(float), sizeof(float));
      data.push_back(value);
    }
  }

  file.close();
}

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
}  // namespace vueron

#endif  // __PCL_H__
