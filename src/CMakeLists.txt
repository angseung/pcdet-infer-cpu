# src/CMakeLists.txt

add_subdirectory(pcdet-infer-cpu)

if(UNIX)
add_subdirectory(draw)
add_executable(demo demo.cpp)
target_include_directories(demo PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(
    demo
    pcdetfunc
    draw
    ${PCL_LIBRARIES}
)
endif()
