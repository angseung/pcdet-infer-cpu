# tests/pcdet_test/CMakeLists.txt

# A PCDet library ONLY for test codes
add_library(pcdet_test STATIC pre.cpp rpn.cpp post.cpp model.cpp ort_model.cpp pcdet.cpp)

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

target_include_directories(pcdet_test PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${ONNXRUNTIME_ROOTDIR}/include
)
target_link_libraries(pcdet_test PUBLIC onnx::runtime)
