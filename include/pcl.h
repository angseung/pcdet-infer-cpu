#ifndef __PCL_H__
#define __PCL_H__

#include <filesystem>  // for std::filesystem
#include <string>
#include <vector>

namespace vueron {

class PCDReader {
 private:
  std::vector<float> data;
  int stride = 0;

 public:
  PCDReader() = delete;
  explicit PCDReader(const std::string &filePath);
  ~PCDReader() = default;
  const std::vector<float> &getData() const noexcept;
  int getStride() const noexcept;
  PCDReader(PCDReader &copy) = delete;
  PCDReader &operator=(PCDReader &copy) = delete;
};

std::vector<std::string> getPCDFileList(const std::string &folder_path);

std::vector<std::string> getFileList(const std::string &folder_path);
}  // namespace vueron

#endif  // __PCL_H__
