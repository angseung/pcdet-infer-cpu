#ifndef __TYPE_H__
#define __TYPE_H__
#include <cstddef>  // for size_t

#include "params.h"

namespace vueron {
struct BndBox {
  float x, y, z, dx, dy, dz, heading;
};

struct PredBox {
  float x, y, z, dx, dy, dz, heading, score, label;
};

struct Pillar {
  size_t point_index[MAX_NUM_POINTS_PER_PILLAR] = {0};
  size_t pillar_grid_x = 0;
  size_t pillar_grid_y = 0;
  size_t point_num_in_pillar = 0;
  bool is_empty = true;
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
}  // namespace vueron
#endif  // __TYPE_H__
