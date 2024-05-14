#include <iostream>
#include "foo.h"
#include "shape.h"

int main() {
    std::cout << "Hello, CMake" << std::endl;
    std::cout << "Foo : " << foo() << std::endl;

    vueron::Rectangle rect = vueron::Rectangle(10, 20);
    std::cout << rect.GetSize() << std::endl;

    vueron::Rectangle rect2 = vueron::Rectangle();
    std::cout << "by Defaults\n" << rect2.GetSize() << std::endl;

    return 0;
}
