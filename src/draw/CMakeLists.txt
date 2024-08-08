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
    
    # Find OpenMP
    find_package(OpenMP REQUIRED)

    if (OpenMP_FOUND)
        message(STATUS "Found OpenMP")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else ()
        message(FATAL_ERROR "Could not find OpenMP")
    endif ()
endif ()

find_package(OpenCV REQUIRED)

add_library(draw STATIC draw.cpp)
target_include_directories(draw PUBLIC ${PROJECT_SOURCE_DIR}/include)
if (ENABLE_OPEN3D)
    target_link_libraries(draw PUBLIC opencv_core opencv_imgproc opencv_highgui Open3D::Open3D)
else ()
    target_link_libraries(draw PUBLIC opencv_core opencv_imgproc opencv_highgui)
endif ()
