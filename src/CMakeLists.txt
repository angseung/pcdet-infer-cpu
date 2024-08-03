# src/CMakeLists.txt

add_subdirectory(pcl)
add_subdirectory(pcdet-infer-cpu)

if (BUILD_DEMO)
    add_subdirectory(draw)
    add_executable(demo demo.cpp)
    target_link_libraries(
            demo
            pcl
            pcdet
            draw
    )
    add_executable(demo_c demo_c.cpp)
    target_link_libraries(
            demo_c
            pcl
            pcdet
            draw
    )
    if (ENABLE_OPEN3D)
        add_executable(demo_3d demo_3d.cpp)
        target_link_libraries(
                demo_3d
                pcl
                pcdet
                draw
        )
    endif ()
else ()
    message(STATUS "SKIP BUILD DEMO")
endif ()
