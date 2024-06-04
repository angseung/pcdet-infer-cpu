#ifndef __TYPE_H__
#define __TYPE_H__

namespace vueron {
struct BndBox {
  float x, y, z, dx, dy, dz, heading;
};

struct PredBox {
  float x, y, z, dx, dy, dz, heading, score, label;
};
}  // namespace vueron
#endif  // __TYPE_H__
