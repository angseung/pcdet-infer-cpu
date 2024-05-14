#include <iostream>
#include <thread>

#include "shape.h"
namespace vueron
{
    Rectangle::Rectangle(int width, int height) : width_(width), height_(height) {}
    Rectangle::Rectangle(){
        width_ = 1;
        height_ = 5;
    }
    Rectangle::~Rectangle() {};

    int Rectangle::GetSize() const {
    std::thread t([this]() { std::cout << "Calulate .." << std::endl; });
    t.join();

    // 직사각형의 넓이를 리턴한다.
    return width_ * height_;
    }
}
