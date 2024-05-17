# src/CMakeLists.txt

add_library(pcdet STATIC pcl.cpp snapshot.cpp pre.cpp)

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

target_include_directories(pcdet PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/includes)
# target_compile_options(pcdet PRIVATE -Wall -Werror)
target_link_libraries(pcdet ${PCL_LIBRARIES})
