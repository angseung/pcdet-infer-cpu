#include "params.h"
#include "pcl.h"
#include "shape.h"
#include <glob.h>
#include <iostream>

int main(int argc, const char **argv) {
    std::string folder_path = PCD_PATH;
    std::vector<std::string> pcd_files = getPcdFiles(folder_path);
    std::vector<float> points;

    for (const auto &file : pcd_files) {
        std::cout << file << std::endl;
        points = readPcdFile(file, MAX_POINTS_NUM);
        std::cout << "Points Num of " << file << ": "
                  << points.size() / sizeof(float) << std::endl;
    }

    std::cout << "Hello, CMake" << std::endl;

    vueron::Rectangle rect = vueron::Rectangle(10, 20);
    std::cout << rect.GetSize() << std::endl;

    vueron::Rectangle rect2 = vueron::Rectangle();
    std::cout << "by Defaults\n" << rect2.GetSize() << std::endl;

    return 0;
}
