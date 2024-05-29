# src/CMakeLists.txt

add_library(pcdet STATIC pcl.cpp snapshot.cpp pre.cpp rpn.cpp post.cpp)

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

if(APPLE)
set(ONNXRUNTIME_ROOTDIR ${PROJECT_SOURCE_DIR}/onnxruntime-osx-arm64-1.16.3)
add_library(_onnxruntime INTERFACE IMPORTED)
set_target_properties(_onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOTDIR}/include
    INTERFACE_LINK_LIBRARIES ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.dylib
)
elseif(UNIX)
set(ONNXRUNTIME_ROOTDIR ${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.16.3)
add_library(_onnxruntime INTERFACE IMPORTED)
set_target_properties(_onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOTDIR}/include
    INTERFACE_LINK_LIBRARIES ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so
)
else()
message(FATAL_ERROR "Unsupported OS")
endif()

add_library(onnx::runtime ALIAS _onnxruntime)

target_include_directories(pcdet PUBLIC 
    ${PROJECT_SOURCE_DIR}/src 
    ${PROJECT_SOURCE_DIR}/includes 
    ${ONNXRUNTIME_ROOTDIR}/include
)
target_link_libraries(pcdet PUBLIC ${PCL_LIBRARIES} onnx::runtime)

add_subdirectory(draw)
add_executable(demo demo.cpp)
target_include_directories(demo PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/includes)
target_link_libraries(
    demo
    pcdet
    draw 
    ${PCL_LIBRARIES}
)
