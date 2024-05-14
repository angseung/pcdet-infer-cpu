enable_testing()

add_executable(
  hello_test
  hello_test.cpp
  area_test.cpp
)
target_link_libraries(
  hello_test
  GTest::gtest_main
  shape
)

include(GoogleTest)
gtest_discover_tests(hello_test)
