cmake_minimum_required(VERSION 3.11)

project(
        pcdet-infer-cpu
        VERSION 0.1
        DESCRIPTION "cpu inference framework for pcdet models"
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build Options
set(BUILD_DEMO ON CACHE BOOL "Build Demo")
set(BUILD_MAIN ON CACHE BOOL "Build Main")
set(BUILD_TEST ON CACHE BOOL "Build Tests")
set(ENABLE_OPEN3D OFF CACHE BOOL "Build emo_3d")

# Logging Options
set(_PROFILE OFF CACHE BOOL "Logging Model Latency")
set(_DEBUG OFF CACHE BOOL "Logging Debug Messages")

set(PFE_FILE ${PROJECT_SOURCE_DIR}/models/gcm_v4_residual/gcm_resfpn_falcon_v4_residual_gelu_full_zero_f71.68_r0_s52.48_pfe.onnx)
set(RPN_FILE ${PROJECT_SOURCE_DIR}/models/gcm_v4_residual/gcm_resfpn_falcon_v4_residual_gelu_full_zero_f71.68_r0_s52.48_rpn.onnx)
set(PCD_PATH ${PROJECT_SOURCE_DIR}/pcd/cepton)

# Configure version.h
message(STATUS "Generating version.h in include directory...")
execute_process(
        COMMAND sh version.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
message(STATUS "Generating done.")

# Configure config header
configure_file(
        "${PROJECT_SOURCE_DIR}/include/config.in"
        "${PROJECT_SOURCE_DIR}/include/config.h"
        @ONLY
)

# Set the output directory for executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# pcdet & demo
add_subdirectory(src)

# main
if (BUILD_MAIN)
    add_executable(main main.cpp)
    target_link_libraries(main PUBLIC pcl pcdet)
else ()
    message(STATUS "SKIP BUILD MAIN")
endif ()

# GoogleTest
if (BUILD_TEST)
    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Configure config header for tests
    set(SNAPSHOT_PATH ${PROJECT_SOURCE_DIR}/zero_snapshots/pcd*)
    configure_file(
            "${PROJECT_SOURCE_DIR}/tests/pcdet_test/include/params.in"
            "${PROJECT_SOURCE_DIR}/tests/pcdet_test/include/params.h"
            @ONLY
    )
    configure_file(
            "${PROJECT_SOURCE_DIR}/tests/pcdet_test/include/config.in"
            "${PROJECT_SOURCE_DIR}/tests/pcdet_test/include/config.h"
            @ONLY
    )
    add_subdirectory(tests)
else ()
    message(STATUS "SKIP BUILD TESTS")
endif ()
