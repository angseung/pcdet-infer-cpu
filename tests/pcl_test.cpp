#include "pcl.h"

#include <glob.h>
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "npy.h"
#include "pcdet_test/include/config.h"
#include "pcdet_test/include/params.h"
#include "pcl.h"
#include "type.h"

#define _ERROR 1e-3

TEST(IntegrationTest, IntegrationTest) {
  std::string folder_path = PCD_PATH;
  std::vector<std::string> pcd_files = vueron::getPCDFileList(folder_path);
  const size_t num_test_files = pcd_files.size();
  constexpr size_t point_stride = POINT_STRIDE;

  for (size_t i = 0; i < num_test_files; i++) {
    const std::string pcd_file = pcd_files[i];
    std::cout << "Testing : " << pcd_file << std::endl;
    const std::vector<float> buffer =
        vueron::readPcdFile(pcd_file, MAX_POINT_NUM);
    const float* points = buffer.data();
    size_t point_buf_len = buffer.size();

    vueron::PCDReader reader(pcd_file, MAX_POINT_NUM);
    const std::vector<float> buffer_new = reader.getData();
    const size_t point_stride_new = reader.getStride();
    const float* points_new = buffer_new.data();

    EXPECT_EQ(buffer_new.size() / point_stride_new,
              buffer.size() / POINT_STRIDE);

    for (size_t i = 0; i < buffer.size() / POINT_STRIDE; i++) {
      EXPECT_EQ(points[POINT_STRIDE * i], points_new[point_stride_new * i]);
      EXPECT_EQ(points[POINT_STRIDE * i + 1],
                points_new[point_stride_new * i + 1]);
      EXPECT_EQ(points[POINT_STRIDE * i + 2],
                points_new[point_stride_new * i + 2]);
      EXPECT_EQ(points[POINT_STRIDE * i + 3],
                points_new[point_stride_new * i + 3]);
    }

    std::cout << "Test Finish : " << pcd_file << std::endl;
  }
}
