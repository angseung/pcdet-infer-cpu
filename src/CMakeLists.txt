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
else ()
    message(STATUS "SKIP BUILD DEMO")
endif ()
