# src/pcdet-cpu/CMakeLists.txt

add_subdirectory(common)

# onnx::runtime
if (APPLE)
    set(ONNXRUNTIME_ROOTDIR ${PROJECT_SOURCE_DIR}/onnxruntime-osx-arm64-1.16.3)
    add_library(_onnxruntime INTERFACE IMPORTED)
    set_target_properties(_onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOTDIR}/include
            INTERFACE_LINK_LIBRARIES ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.dylib
    )
elseif (UNIX)
    set(ONNXRUNTIME_ROOTDIR ${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.16.3)
    add_library(_onnxruntime INTERFACE IMPORTED)
    set_target_properties(_onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOTDIR}/include
            INTERFACE_LINK_LIBRARIES ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so
    )
else ()
    message(FATAL_ERROR "Unsupported OS")
endif ()
add_library(onnx::runtime ALIAS _onnxruntime)

set(PCDET_SRC_FILES pre.cpp post.cpp ort_model.cpp pcdet.cpp pcdet_c.cpp)
add_library(pcdet STATIC ${PCDET_SRC_FILES})
target_include_directories(
        pcdet
        PUBLIC
        ${ONNXRUNTIME_ROOTDIR}/include
)
target_link_libraries(pcdet PUBLIC json_config onnx::runtime)
