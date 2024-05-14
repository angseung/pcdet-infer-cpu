# /lib/CMakeLists.txt

add_library(shape STATIC shape.cpp)
target_include_directories(shape PUBLIC ${CMAKE_SOURCE_DIR}/includes)
target_compile_options(shape PRIVATE -Wall -Werror)
target_link_libraries(shape PRIVATE pthread)
