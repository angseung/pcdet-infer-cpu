# src/draw/CMakeLists.txt
if (ENABLE_OPEN3D)
    if (APPLE)
        list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/open3d-devel-darwin-arm64-0.18.0/lib/cmake)
    elseif (UNIX)
        list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/open3d-devel-linux-x86_64-cxx11-abi-0.18.0/lib/cmake)
    else ()
        message(FATAL_ERROR "Unsupported OS")
    endif ()
    find_package(Open3D REQUIRED)
endif ()

find_package(OpenCV REQUIRED)

add_library(draw STATIC draw.cpp)
set(DRAW_OBJS_LIST opencv_core opencv_imgproc opencv_highgui)
target_include_directories(draw PUBLIC ${PROJECT_SOURCE_DIR}/include)
if (ENABLE_OPEN3D)
    list(APPEND DRAW_OBJS_LIST Open3D::Open3D)
endif ()

target_link_libraries(draw PUBLIC ${DRAW_OBJS_LIST})
