# tests/CMakeLists.txt
add_subdirectory(pcdet_test)
enable_testing()
set(TEST_COMMON_OBJ
        GTest::gtest_main
        pcdet_test
        pcl
)

add_executable(
        voxel_snapshot_test
        snapshot_shape_test.cpp
)
target_link_libraries(voxel_snapshot_test ${TEST_COMMON_OBJ})

add_executable(
        preprocess_value_test
        snapshot_value_test.cpp
)
target_link_libraries(preprocess_value_test ${TEST_COMMON_OBJ})

add_executable(
        rpn_test
        snaptshot_rpn_test.cpp
)
target_link_libraries(rpn_test ${TEST_COMMON_OBJ})

add_executable(
        integration_test
        integration_test.cpp
)
target_link_libraries(integration_test ${TEST_COMMON_OBJ})

add_executable(
        class_test
        class_test.cpp
)
target_link_libraries(class_test ${TEST_COMMON_OBJ})

include(GoogleTest)
gtest_discover_tests(voxel_snapshot_test)
gtest_discover_tests(preprocess_value_test)
gtest_discover_tests(rpn_test)
gtest_discover_tests(integration_test)
gtest_discover_tests(class_test)
