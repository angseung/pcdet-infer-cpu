# src/pcdet-infer-cpu/common/CMakeLists.txt
add_library(json_config STATIC metadata.cpp runtimeconfig.cpp)

# Add the PIC flag
set_target_properties(json_config PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Json library
include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
CPMAddPackage("gh:nlohmann/json@3.10.5")

target_include_directories(
        json_config
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(json_config PUBLIC nlohmann_json::nlohmann_json)
