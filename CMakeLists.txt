cmake_minimum_required(VERSION 3.11)

project(
  pcdet-infer-onnx
  VERSION 0.1
  DESCRIPTION "cpu inference framework for pcdet models"
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
# Set the output directory for executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable (test main.cpp foo.cpp)
target_include_directories(test PUBLIC ${CMAKE_SOURCE_DIR}/includes)
target_compile_options(test PUBLIC -Wall -Werror)
add_subdirectory(lib)
target_link_libraries(test PUBLIC shape)
