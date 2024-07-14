#ifndef __TYPE_H__
#define __TYPE_H__

namespace vueron {

struct BndBox {
  float x, y, z, dx, dy, dz, heading;

  // Operator overload to cast BndBox to const float*
  explicit operator const float*() const {
    return &x;  // Return the address of the 'x' member
  }
  explicit operator float() const = delete;
};

struct PredBox {
  float x, y, z, dx, dy, dz, heading, score, label;

  // Operator overload to cast BndBox to const float*
  explicit operator const float*() const {
    return &x;  // Return the address of the 'x' member
  }
  explicit operator float() const = delete;
};
}  // namespace vueron

#endif  // __TYPE_H__
