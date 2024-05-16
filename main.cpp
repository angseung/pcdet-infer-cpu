#include "params.h"
#include "pcl.h"
#include "pre.h"
#include "shape.h"
#include <glob.h>
#include <iostream>

int main(int argc, const char **argv) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = getPcdFiles(folder_path);
    std::vector<float> points;

    for (const auto &file : pcd_files) {
        points = readPcdFile(file, MAX_POINTS_NUM);
#ifdef _DEBUG
        std::cout << file << std::endl;
        std::cout << "Points Num of " << file << ": "
                  << points.size() / sizeof(float) << std::endl;
#endif
        std::vector<vueron::Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
        vueron::voxelization(bev_pillar, (float *)points.data(), points.size(),
                             sizeof(float));
    }

    return 0;
}
