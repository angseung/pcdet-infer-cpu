#include <gtest/gtest.h>
#include "shape.h"

// Demonstrate some basic assertions.
TEST(AreaTest, BasicAssertions) {
  // Expect equality.
  vueron::Rectangle rect = vueron::Rectangle(10, 4);
  int area = rect.GetSize();
  EXPECT_EQ(area, 40);
}
