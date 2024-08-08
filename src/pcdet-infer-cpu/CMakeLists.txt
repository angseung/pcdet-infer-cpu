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
set(PCDET_VERSION 1.0.0 CACHE STRING "PCDet version")
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

# zip library files
#set(PCDET_LIB_FILES
#        ${PROJECT_SOURCE_DIR}/include/pcdet-infer-cpu/pcdet_c.h
#        ${PROJECT_SOURCE_DIR}/include/pcdet-infer-cpu/common/runtimeconfig.h
#        ${PROJECT_SOURCE_DIR}/include/pcdet-infer-cpu/common/metadata.h
#        ${PROJECT_SOURCE_DIR}/include/pcdet-infer-cpu/common/type.h
#)
#if (APPLE)
#    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.${PCDET_VERSION}.dylib)
#    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.${PCDET_SO_VERSION}.dylib)
#    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.dylib)
#elseif (UNIX)
#    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.${PCDET_VERSION}.so)
#    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.${PCDET_SO_VERSION}.so)
#    list(APPEND PCDET_LIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libpcdet_shared.so)
#else ()
#    message(FATAL_ERROR "Unsupported OS. This library supports only macOS & linux ONLY")
#endif ()
#
#set(OS_NAME ${CMAKE_SYSTEM_NAME})
#set(ARCH_NAME ${CMAKE_SYSTEM_PROCESSOR})
#set(TAR_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pcdet_${PCDET_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}.tar.gz)
#add_custom_command(
#        OUTPUT ${TAR_OUTPUT}
#        COMMAND ${CMAKE_COMMAND} -E tar zcvf ${TAR_OUTPUT} ${PCDET_LIB_FILES}
#        DEPENDS ${PCDET_LIB_FILES}
#        COMMENT "Generating tar.gz archive of pcdet library"
#)
#add_custom_target(create_tar ALL DEPENDS ${TAR_OUTPUT})
