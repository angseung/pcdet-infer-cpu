enable_testing()

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
)

include(GoogleTest)
gtest_discover_tests(voxel_snapshot_test)
gtest_discover_tests(preprocess_value_test)
