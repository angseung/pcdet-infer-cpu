# src/pcdet-infer-cpu/common/CMakeLists.txt

add_library(json_config STATIC metadata.cpp runtimeconfig.cpp)

target_include_directories(
        json_config
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)
