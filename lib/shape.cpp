#include "shape.h"
#include <iostream>
#include <thread>

namespace vueron {
Rectangle::Rectangle(int width, int height) : width_(width), height_(height){};
Rectangle::Rectangle() {
    width_ = 1;
    height_ = 5;
}
Rectangle::~Rectangle(){};

int Rectangle::GetSize() const { return width_ * height_; }
} // namespace vueron
