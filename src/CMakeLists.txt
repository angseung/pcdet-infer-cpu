# src/CMakeLists.txt

add_subdirectory(pcdet-infer-cpu)
add_subdirectory(pcl)

if (${BUILD_DEMO})
    add_subdirectory(draw)
    add_executable(demo demo.cpp)
    target_link_libraries(
            demo
            pcdet
            draw
            pcl
    )
else ()
    message(STATUS "SKIP BUILD DEMO")
endif ()
