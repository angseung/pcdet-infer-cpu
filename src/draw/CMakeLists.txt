find_package(OpenCV REQUIRED)

add_library(draw STATIC
  draw.cpp
)
target_include_directories(draw PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(draw PUBLIC
  pcdet
  opencv_core opencv_imgproc opencv_highgui
)
