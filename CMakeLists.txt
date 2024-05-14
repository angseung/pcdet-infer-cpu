cmake_minimum_required(VERSION 3.11)

project(
  pcdet-infer-onnx
  VERSION 0.1
  DESCRIPTION "cpu inference framework for pcdet models"
  LANGUAGES C CXX)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Set the output directory for executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable (test main.cpp)
add_subdirectory(tests)
add_subdirectory(lib)
target_include_directories(test PUBLIC ${CMAKE_SOURCE_DIR}/includes)
target_compile_options(test PUBLIC -Wall -Werror)
target_link_libraries(test PUBLIC shape)
