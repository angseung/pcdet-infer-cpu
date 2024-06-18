# src/CMakeLists.txt

add_subdirectory(pcdet-infer-cpu)

if (${BUILD_DEMO})
    add_subdirectory(draw)
    add_executable(demo demo.cpp)
    target_link_libraries(
            demo
            pcdet
            draw
    )
else ()
    message(STATUS "SKIP BUILD DEMO")
endif ()
