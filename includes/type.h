// I avoided to use #pragma once for c
#ifndef BOX_H
#define BOX_H

namespace vueron {
// TODO: float -> double or boosted int16_t value
// TODO: should I pad it?
struct BndBox {
    float x, y, z, dx, dy, dz, heading; // , _ /*pad*/;
};
} // namespace vueron
#endif // BOX_H
