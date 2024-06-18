# src/pcl/CMakeLists.txt

add_library(pcl STATIC pcl.cpp)
target_include_directories(pcl PRIVATE ${PROJECT_SOURCE_DIR}/include)
