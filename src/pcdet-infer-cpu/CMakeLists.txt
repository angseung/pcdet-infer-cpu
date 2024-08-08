# src/pcdet-cpu/CMakeLists.txt

# metadata & runtimeconfig
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
    message(FATAL_ERROR "Unsupported OS. This library supports only macOS & linux ONLY")
endif ()

add_library(onnx::runtime ALIAS _onnxruntime)

set(PCDET_SRC_FILES pre.cpp post.cpp ort_model.cpp pcdet.cpp pcdet_c.cpp)
set(PCDET_OBJS json_config onnx::runtime)
add_library(pcdet STATIC ${PCDET_SRC_FILES})
target_include_directories(
        pcdet
        PUBLIC
        ${ONNXRUNTIME_ROOTDIR}/include
)
target_link_libraries(pcdet PUBLIC ${PCDET_OBJS})

# Shared Object for pcdet
set(PCDET_VERSION 1.4.0 CACHE STRING "PCDet version")
set(PCDET_SO_VERSION 1)
add_library(pcdet_shared SHARED ${PCDET_SRC_FILES})
set_target_properties(pcdet_shared PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set_target_properties(pcdet_shared PROPERTIES OUTPUT_NAME pcdet_shared VERSION ${PCDET_VERSION} SOVERSION ${PCDET_SO_VERSION})
target_include_directories(
        pcdet_shared
        PUBLIC
        ${ONNXRUNTIME_ROOTDIR}/include
)
target_link_libraries(pcdet_shared PUBLIC ${PCDET_OBJS})

## zip library files
# 1. Create target directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet/pcdet)

# 2. Copy header files to the target directory
set(PCDET_HRADER_FILES
        ${PROJECT_SOURCE_DIR}/include/pcdet-infer-cpu/pcdet_c.h
        ${PROJECT_SOURCE_DIR}/include/pcdet-infer-cpu/common/runtimeconfig.h
        ${PROJECT_SOURCE_DIR}/include/pcdet-infer-cpu/common/metadata.h
        ${PROJECT_SOURCE_DIR}/include/pcdet-infer-cpu/common/type.h
)
add_custom_command(
        TARGET pcdet_shared POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet
        COMMAND ${CMAKE_COMMAND} -E copy ${PCDET_HRADER_FILES} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet/pcdet
        COMMENT "Copying PCDet headers after build."
)

# 3. Copy lib files to the target directory
set(PCDET_LIB_FILES)
if (APPLE)
    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.${PCDET_VERSION}.dylib)
    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.${PCDET_SO_VERSION}.dylib)
    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.dylib)
elseif (UNIX)
    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.so.${PCDET_VERSION})
    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.so.${PCDET_SO_VERSION})
    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.so)
else ()
    message(FATAL_ERROR "Unsupported OS. This library supports only macOS & linux ONLY")
endif ()
add_custom_command(
        TARGET pcdet_shared POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet
        COMMAND ${CMAKE_COMMAND} -E copy ${PCDET_LIB_FILES} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet
        COMMENT "Copying PCDet shared libraries after build."
)

# 4. zip target directory
set(OS_NAME ${CMAKE_SYSTEM_NAME})
set(ARCH_NAME ${CMAKE_SYSTEM_PROCESSOR})
set(TAR_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pcdet_${PCDET_VERSION}_${CMAKE_BUILD_TYPE}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}.tar.gz)
add_custom_command(
        TARGET pcdet_shared POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E tar "zcf" ${TAR_OUTPUT} -- ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet
        # delete temporary directory when compression is done.
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet
        DEPENDS ${PCDET_LIB_FILES}
        COMMENT "Creating tar.gz archive in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}."
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
