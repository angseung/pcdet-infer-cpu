# src/draw/CMakeLists.txt

find_package(OpenCV REQUIRED)

add_library(draw STATIC draw.cpp)
target_include_directories(draw PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/includes)
target_link_libraries(draw PUBLIC opencv_core opencv_imgproc opencv_highgui)
