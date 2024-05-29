#ifndef PARAMS_H
#define PARAMS_H

#include <iostream>

// Params for Preprocessing
#define MAX_POINTS_NUM 100000
#define NUM_POINT_VALUES 4
#define INTENSITY_NORMALIZE_DIV 255
#define GRID_X_SIZE 224
#define GRID_Y_SIZE 328
#define MAX_NUM_POINTS_PER_PILLAR 20
#define MAX_VOXELS 25000
#define FEATURE_NUM 10
#define NUM_FEATURE_SCATTER 64
#define ZERO_INTENSITY

#define MIN_X_RANGE 0.0f
#define MAX_X_RANGE 71.68f
#define MIN_Y_RANGE -52.48f
#define MAX_Y_RANGE 52.48f
#define MIN_Z_RANGE -2.0f
#define MAX_Z_RANGE 4.0f

#define VOXEL_X_SIZE 0.32f
#define VOXEL_Y_SIZE 0.32f
#define VOXEL_Z_SIZE 6.0f

// Params for Poseprocessing
#define CLASS_NUM 3
#define IOU_RECTIFIER                                                          \
    { 0.68f, 0.71f, 0.65f }
#define MAX_BOX_NUM_BEFORE_NMS 500
#define MAX_BOX_NUM_AFTER_NMS 83
#define IOU_THRESH 0.2f
#define FEATURE_X_SIZE 112
#define FEATURE_Y_SIZE 164
#define SCORE_THRESH 0.1f

// Params for Drawing
#define CONF_THRESH 0.4f

#endif // PARAMS_H
