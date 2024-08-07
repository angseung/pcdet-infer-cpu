# src/CMakeLists.txt

add_subdirectory(pcl)
add_subdirectory(pcdet-infer-cpu)
set(DEMO_OBJS_COMMON
        pcl
        pcdet
        draw
)

if (BUILD_DEMO)
    add_subdirectory(draw)
    add_executable(demo demo.cpp)
    target_link_libraries(demo ${DEMO_OBJS_COMMON})
    add_executable(demo_c demo_c.cpp)
    target_link_libraries(demo_c ${DEMO_OBJS_COMMON})
    
    if (ENABLE_OPEN3D)
        add_executable(demo_3d demo_3d.cpp)
        target_link_libraries(demo_3d ${DEMO_OBJS_COMMON})
    else ()
        message(STATUS "SKIP BUILD DEMO_3D")
    endif ()
else ()
    message(STATUS "SKIP BUILD DEMO")
endif ()
