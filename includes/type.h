// I avoided to use #pragma once for c
#ifndef __PCDET_BOX_H__
#define __PCDET_BOX_H__

// TODO: float -> double or boosted int16_t value
// TODO: should I pad it?
struct BndBox {
    float x, y, z, dx, dy, dz, heading; // , _ /*pad*/;
};

#endif // __PCDET_BOX_H__
