enable_testing()

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(
  hello_test
  hello_test.cpp
  area_test.cpp
)
target_link_libraries(
  hello_test
  pcdet
  GTest::gtest_main
  shape
)
# target_include_directories(hello_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/includes)

add_executable(
  voxel_snapshot_test
  snapshot_shape_test.cpp
)
target_link_libraries(
  voxel_snapshot_test
  GTest::gtest_main
  pcdet
  ${PCL_LIBRARIES}
)
# target_include_directories(voxel_snapshot_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/includes)

include(GoogleTest)
gtest_discover_tests(hello_test)
gtest_discover_tests(voxel_snapshot_test)
