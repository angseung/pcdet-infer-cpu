# src/draw/CMakeLists.txt
if (APPLE)
    list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/open3d-devel-darwin-arm64-0.18.0+b271acb/lib/cmake)
elseif (UNIX)
    list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/open3d-devel-linux-x86_64-cxx11-abi-0.18.0+b271acb/lib/cmake)
else ()
    message(FATAL_ERROR "Unsupported OS")
endif ()
find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)

add_library(draw STATIC draw.cpp)
target_include_directories(draw PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(draw PUBLIC opencv_core opencv_imgproc opencv_highgui Open3D::Open3D)
