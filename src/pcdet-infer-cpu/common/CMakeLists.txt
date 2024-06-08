# src/pcdet-infer-cpu/common/CMakeLists.txt

add_library(json_config STATIC metadata.cpp runtimeconfig.cpp)

target_include_directories(json_config PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

# Json library
include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
CPMAddPackage("gh:nlohmann/json@3.10.5")

target_link_libraries(json_config PRIVATE nlohmann_json::nlohmann_json)
