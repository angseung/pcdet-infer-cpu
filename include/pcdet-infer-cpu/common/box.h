#ifndef __TYPE_H__
#define __TYPE_H__

struct Box {
  float x, y, z, dx, dy, dz, heading;

#ifdef __cplusplus
  // Operator overload to cast Box to const float*
  explicit operator const float*() const {
    return &x;  // Return the address of the 'x' member
  }
  explicit operator float() const = delete;
#endif  // __cplusplus
};

struct Bndbox {
  float x;
  float y;
  float z;
  float dx;
  float dy;
  float dz;
  float heading;
  int label;
  float score;

#ifdef __cplusplus
  // Operator overload to cast Bndbox to const float*
  explicit operator const float*() const {
    return &x;  // Return the address of the 'x' member
  }
  explicit operator float() const = delete;
#endif  // __cplusplus
};

#endif  // __TYPE_H__
