cmake_minimum_required(VERSION 3.11)

project(
  pcdet-infer-cpu
  VERSION 0.1
  DESCRIPTION "cpu inference framework for pcdet models"
  LANGUAGES CXX)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(PFE_PATH ${PROJECT_SOURCE_DIR}/models/gcm_resfpn_falcon_v4_residual_gelu_full_zero_f71.68_r0_s52.48_pfe.onnx)
set(RPN_PATH ${PROJECT_SOURCE_DIR}/models/gcm_resfpn_falcon_v4_residual_gelu_full_zero_f71.68_r0_s52.48_rpn.onnx)
set(PCD_PATH ${PROJECT_SOURCE_DIR}/pcd/*.pcd)
set(SNAPSHOT_PATH ${PROJECT_SOURCE_DIR}/zero_snapshots/pcd*)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/includes/config.in"
  "${PROJECT_SOURCE_DIR}/includes/config.h"
  @ONLY
  )

# Set the output directory for executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_executable(main main.cpp)
add_subdirectory(tests)
target_link_libraries(main PUBLIC pcdetfunc)
