#ifndef __TYPE_H__
#define __TYPE_H__

namespace vueron {
struct BndBox {
    float x, y, z, dx, dy, dz, heading;
};

struct Point {
    float x{}, y{};
    Point() = default;
    Point(const float _x, const float _y) { x = _x, y = _y; }

    void set(const float _x, const float _y) {
        x = _x;
        y = _y;
    }

    Point operator+(const Point &b) const { return {x + b.x, y + b.y}; }

    Point operator-(const Point &b) const { return {x - b.x, y - b.y}; }
};
} // namespace vueron
#endif // __TYPE_H__
