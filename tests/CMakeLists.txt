enable_testing()

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

add_executable(
  voxel_snapshot_test
  snapshot_shape_test.cpp
)
target_link_libraries(
  voxel_snapshot_test
  GTest::gtest_main
  pcdet
  shape
)

add_executable(
  preprocess_value_test
  snapshot_value_test.cpp
)
target_link_libraries(
  preprocess_value_test
  GTest::gtest_main
  pcdet
  shape
)

include(GoogleTest)
gtest_discover_tests(hello_test)
gtest_discover_tests(voxel_snapshot_test)
gtest_discover_tests(preprocess_value_test)
