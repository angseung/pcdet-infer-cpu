cmake_minimum_required(VERSION 3.11)

project(
	pcdet-infer-cpu
	VERSION 0.1
	DESCRIPTION "cpu inference framework for pcdet models"
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build Options
set(BUILD_DEMO ON CACHE BOOL "Build Demo")
set(BUILD_TEST ON CACHE BOOL "Build Tests")

# Logging Options
set(_PROFILE OFF CACHE BOOL "Logging Model Latency")
set(_DEBUG OFF CACHE BOOL "Logging Debug Messages")

# Point cloud library
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# GoogleTest
if(${BUILD_TEST})
include(FetchContent)
FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
endif()

set(PFE_PATH ${PROJECT_SOURCE_DIR}/models/gcm_v4_residual/gcm_resfpn_falcon_v4_residual_gelu_full_zero_f71.68_r0_s52.48_pfe.onnx)
set(RPN_PATH ${PROJECT_SOURCE_DIR}/models/gcm_v4_residual/gcm_resfpn_falcon_v4_residual_gelu_full_zero_f71.68_r0_s52.48_rpn.onnx)
set(PCD_PATH ${PROJECT_SOURCE_DIR}/pcd/cepton/*.pcd)
set(SNAPSHOT_PATH ${PROJECT_SOURCE_DIR}/zero_snapshots/pcd*)

# Configure config header
configure_file (
	"${PROJECT_SOURCE_DIR}/include/config.in"
	"${PROJECT_SOURCE_DIR}/include/config.h"
	@ONLY
)

# Set the output directory for executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_executable(main main.cpp)
target_link_libraries(main PUBLIC pcdet ${PCL_LIBRARIES})

if(${BUILD_TEST})
add_subdirectory(tests)
else()
message(STATUS "SKIP BUILD TESTS")
endif()
