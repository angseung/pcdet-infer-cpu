#include "npy.h"
#include "params.h"
#include "pcl.h"
#include "pre.h"
#include "shape.h"
#include <glob.h>
#include <iostream>

int main(int argc, const char **argv) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = vueron::getFileList(folder_path);
    std::vector<float> points;

    while (1) {

        for (const auto &file : pcd_files) {
            points = vueron::readPcdFile(file, MAX_POINTS_NUM);
#ifdef _DEBUG
            std::cout << file << std::endl;
            std::cout << "Points Num of " << file << ": "
                      << points.size() / sizeof(float) << std::endl;
#endif
            vueron::preprocess((float *)points.data(), points.size(),
                               sizeof(float));
        }

        // read snapshot
        std::string snapshot_folder_path = SNAPSHOT_PATH;
        std::vector<std::string> snapshot_files =
            vueron::getFileList(snapshot_folder_path);

        for (std::string snapshot_dir : snapshot_files) {
            const std::string voxels_path = snapshot_dir + "/voxels.npy";
            const std::string voxel_coord_path =
                snapshot_dir + "/voxel_coord.npy";
            const std::string voxel_num_points_path =
                snapshot_dir + "/voxel_num_points.npy";
            auto raw_voxels = npy::read_npy<float>(voxels_path);
            auto raw_voxel_coord = npy::read_npy<uint32_t>(voxel_coord_path);
            auto raw_voxel_num_points =
                npy::read_npy<uint32_t>(voxel_num_points_path);

            std::vector<float> voxels = raw_voxels.data;
            std::vector<uint32_t> voxel_coord = raw_voxel_coord.data;
            std::vector<uint32_t> voxel_num_points = raw_voxel_num_points.data;

            std::vector<unsigned long> voxel_shape = raw_voxels.shape;
            std::vector<unsigned long> voxel_coord_shape =
                raw_voxel_coord.shape;
            std::vector<unsigned long> voxel_num_points_shape =
                raw_voxel_num_points.shape;
        }
    }

    return 0;
}
