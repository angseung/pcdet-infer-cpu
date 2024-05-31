# src/CMakeLists.txt

add_subdirectory(pcdet-cpu)

if(UNIX)
add_subdirectory(draw)
add_executable(demo demo.cpp)
target_include_directories(demo PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/includes)
target_link_libraries(
    demo
    pcdetfunc
    draw
    ${PCL_LIBRARIES}
)
endif()
