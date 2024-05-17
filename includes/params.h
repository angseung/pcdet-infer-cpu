#include <iostream>

// common
#define _EPSILON 1e-4
#define OFF 0
#define ON 1
#define _SHIFFLE OFF
// #define _DEBUG

#define PCD_PATH "./pcd/*.pcd"
#define SNAPSHOT_PATH "./snapshots/pcd*"
#define RANDOM_SEED 123

// Params for Point Cloud
#define MAX_POINTS_NUM 100000
#define NUM_POINT_VALUES 4
#define INTENSITY_NORMALIZE_DIV 255

// Params for Voxelization
#define GRID_X_SIZE 224
#define GRID_Y_SIZE 328
#define MAX_NUM_POINTS_PER_PILLAR 20
#define MAX_VOXELS 25000
#define FEATURE_NUM 10
// #define ZERO_INTENSITY

#define MIN_X_RANGE 0.0f
#define MAX_X_RANGE 71.68f
#define MIN_Y_RANGE -52.48f
#define MAX_Y_RANGE 52.48f
#define MIN_Z_RANGE -2.0f
#define MAX_Z_RANGE 4.0f

#define VOXEL_X_SIZE 0.32f
#define VOXEL_Y_SIZE 0.32f
#define VOXEL_Z_SIZE 6.0f
