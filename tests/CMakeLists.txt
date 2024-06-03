# tests/CMakeLists.txt
enable_testing()

add_executable(
	voxel_snapshot_test
	snapshot_shape_test.cpp
)
target_link_libraries(
	voxel_snapshot_test
	GTest::gtest_main
	pcdetfunc
	${PCL_LIBRARIES}
)

add_executable(
	preprocess_value_test
	snapshot_value_test.cpp
)
target_link_libraries(
	preprocess_value_test
	GTest::gtest_main
	pcdetfunc
	${PCL_LIBRARIES}
)

add_executable(
	rpn_test
	snaptshot_rpn_test.cpp
)
target_link_libraries(
	rpn_test
	GTest::gtest_main
	pcdetfunc
	${PCL_LIBRARIES}
)

add_executable(
	integration_test
	integration_test.cpp
)
target_link_libraries(
	integration_test
	GTest::gtest_main
	pcdetfunc
	${PCL_LIBRARIES}
)

add_executable(
	class_test
	class_test.cpp
)
target_link_libraries(
	class_test
	GTest::gtest_main
	pcdet
	${PCL_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(voxel_snapshot_test)
gtest_discover_tests(preprocess_value_test)
gtest_discover_tests(rpn_test)
gtest_discover_tests(integration_test)
gtest_discover_tests(class_test)
