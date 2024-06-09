# src/pcdet-cpu/CMakeLists.txt

add_subdirectory(common)

# onnxruntime
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

add_library(pcdet STATIC pre.cpp post.cpp pcdet.cpp ort_model.cpp)
target_include_directories(pcdet PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${ONNXRUNTIME_ROOTDIR}/include
)
target_link_libraries(pcdet PUBLIC json_config onnx::runtime)
