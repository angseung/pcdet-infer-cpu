# src/CMakeLists.txt

add_subdirectory(pcdet-infer-cpu)

if(UNIX)
add_subdirectory(draw)
add_executable(demo demo.cpp)
target_link_libraries(
    demo
    pcdetfunc
    draw
    ${PCL_LIBRARIES}
)
endif()
