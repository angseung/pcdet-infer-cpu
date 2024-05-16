#include "params.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric> // std::iota
#include <random>
#include <vector>

namespace vueron {

struct VoxelCoord {
    size_t _, __, y, x;
};

struct BaseVoxel {
    float x, y, z;
#if NUM_POINT_VALUES >= 4
    float w;
#endif

#if NUM_POINT_VALUES >= 4
    BaseVoxel() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
#else
    BaseVoxel() : x(0.0f), y(0.0f), z(0.0f) {}
#endif
};

struct Voxel {
    float x, y, z;
#if NUM_POINT_VALUES >= 4
    float w;
#endif
    float offset_from_mean_x, offset_from_mean_y, offset_from_mean_z;
    float offset_from_center_x, offset_from_center_y, offset_from_center_z;
};

struct Pillar {
    size_t point_index[MAX_NUM_POINTS_PER_PILLAR] = {0};
    size_t pillar_grid_x = 0;
    size_t pillar_grid_y = 0;
    size_t point_num_in_pillar = 0;
    bool is_empty = true;
};

void voxelization(std::vector<Pillar> &bev_pillar, const float *points,
                  size_t points_buf_len, size_t point_stride) {
    // check grid size
    assert(GRID_X_SIZE == (float)((MAX_X_RANGE - MIN_X_RANGE) / VOXEL_X_SIZE));
    assert(GRID_Y_SIZE == (float)((MAX_Y_RANGE - MIN_Y_RANGE) / VOXEL_Y_SIZE));
    assert(1 == (float)((MAX_Z_RANGE - MIN_Z_RANGE) / VOXEL_Z_SIZE));

    // check buffer size
    assert(points_buf_len % point_stride == 0);

    std::mt19937 rng(RANDOM_SEED);

    size_t points_num = points_buf_len / point_stride;
    std::vector<size_t> indices(points_num, 0);
    std::vector<BaseVoxel> shuffled(points_num);
    std::iota(indices.begin(), indices.end(), 0);

#if _SHIFFLE == ON
    std::shuffle(indices.begin(), indices.end(), rng);
#endif

    for (size_t i : indices) {
        if (i > MAX_POINTS_NUM)
            break;
        float point_x = points[point_stride * i];
        float point_y = points[point_stride * i + 1];
        float point_z = points[point_stride * i + 2];

#if NUM_POINT_VALUES >= 4
        float point_i = points[point_stride * i + 3];
#endif

        size_t voxel_id_x = floorf((point_x - MIN_X_RANGE) / VOXEL_X_SIZE);
        size_t voxel_id_y = floorf((point_y - MIN_Y_RANGE) / VOXEL_Y_SIZE);

        // skip if out-of-range point
        if (point_x < MIN_X_RANGE || point_x > MAX_X_RANGE ||
            point_y < MIN_Y_RANGE || point_y > MAX_Y_RANGE ||
            point_z < MIN_Z_RANGE || point_z > MAX_Z_RANGE) {
            continue;
        }

        assert(voxel_id_x < GRID_X_SIZE && voxel_id_y < GRID_Y_SIZE);
        size_t voxel_index = voxel_id_y * GRID_X_SIZE + voxel_id_x;

        // bev_pillar : GRID_Y_SIZE * GRID_X_SIZE vector<Pillar>
        if (bev_pillar[voxel_index].point_num_in_pillar < 20) {
            bev_pillar[voxel_index].pillar_grid_x = GRID_X_SIZE;
            bev_pillar[voxel_index].pillar_grid_y = GRID_Y_SIZE;
            bev_pillar[voxel_index]
                .point_index[bev_pillar[voxel_index].point_num_in_pillar] = i;
            bev_pillar[voxel_index].point_num_in_pillar++;
            bev_pillar[voxel_index].is_empty = false;
        }
    }

    size_t count_voxel = 0;

#ifdef _DEBUG
    for (Pillar pillar : bev_pillar) {
        if (!pillar.is_empty) {
            count_voxel++;
        }
    }
    std::cout << "Number of Voxel: " << count_voxel << std::endl;
#endif
}

void voxel_feature_encode(std::vector<Voxel> &voxel,
                          std::vector<BaseVoxel> &base_voxel,
                          const float *points, size_t points_buf_len,
                          size_t point_stride) {}

} // namespace vueron
