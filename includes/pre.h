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
    float x = 0, y = 0, z = 0;
#if NUM_POINT_VALUES >= 4
    float w = 0;
#endif
    float offset_from_mean_x = 0, offset_from_mean_y = 0,
          offset_from_mean_z = 0;
    float offset_from_center_x = 0, offset_from_center_y = 0,
          offset_from_center_z = 0;
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
    size_t processed = 0;

#if _SHIFFLE == ON
    std::shuffle(indices.begin(), indices.end(), rng);
#endif

    for (size_t i : indices) {
        // check max_point_num_per_frame
        if (processed > MAX_POINTS_NUM)
            // requires valiation here
            break;
        float point_x = points[point_stride * i];
        float point_y = points[point_stride * i + 1];
        float point_z = points[point_stride * i + 2];

#if NUM_POINT_VALUES >= 4
        float point_i = points[point_stride * i + 3];
#endif
        processed++;
        assert(processed < MAX_POINTS_NUM);

        size_t voxel_id_x = floorf((point_x - MIN_X_RANGE) / VOXEL_X_SIZE);
        size_t voxel_id_y = floorf((point_y - MIN_Y_RANGE) / VOXEL_Y_SIZE);

        // skip if out-of-range point
        if (point_x < MIN_X_RANGE || point_x > MAX_X_RANGE ||
            point_y < MIN_Y_RANGE || point_y > MAX_Y_RANGE ||
            point_z < MIN_Z_RANGE || point_z > MAX_Z_RANGE) {
            continue;
        }

        // check out-of-range point
        assert(point_x >= MIN_X_RANGE && point_x <= MAX_X_RANGE &&
               point_y >= MIN_Y_RANGE && point_y <= MAX_Y_RANGE &&
               point_z >= MIN_Z_RANGE && point_z <= MAX_Z_RANGE);

        // check out-of-range grid
        assert(voxel_id_x < GRID_X_SIZE && voxel_id_y < GRID_Y_SIZE);

        size_t voxel_index = voxel_id_y * GRID_X_SIZE + voxel_id_x;

        // bev_pillar : GRID_Y_SIZE * GRID_X_SIZE vector<Pillar>
        if (bev_pillar[voxel_index].point_num_in_pillar < 20) {
            size_t voxel_index_in_pillar =
                bev_pillar[voxel_index].point_num_in_pillar;
            bev_pillar[voxel_index].pillar_grid_x = voxel_id_x;
            bev_pillar[voxel_index].pillar_grid_y = voxel_id_y;
            bev_pillar[voxel_index].point_index[voxel_index_in_pillar] = i;
            bev_pillar[voxel_index].point_num_in_pillar++;
            assert(bev_pillar[voxel_index].point_num_in_pillar <=
                   MAX_NUM_POINTS_PER_PILLAR);
            bev_pillar[voxel_index].is_empty = false;
        }
    }

#ifdef _DEBUG
    size_t count_voxel = 0;
    for (Pillar pillar : bev_pillar) {
        if (!pillar.is_empty) {
            count_voxel++;
        }
    }
    std::cout << "Number of Voxel: " << count_voxel << std::endl;
#endif
}

void point_decoration(std::vector<Pillar> &bev_pillar,
                      std::vector<Voxel> &voxels, const float *points,
                      size_t points_buf_len, size_t point_stride) {
    for (Pillar pillar : bev_pillar) {
        if (pillar.is_empty) {
            continue;
        }
        if (pillar.is_empty) {
            continue;
        }
        // calc mean values for all points in current pillar
        float mean_x = 0.0f;
        float mean_y = 0.0f;
        float mean_z = 0.0f;

        // double check grid index of current pillar
        assert(pillar.pillar_grid_x < GRID_X_SIZE &&
               pillar.pillar_grid_y < GRID_Y_SIZE);

        for (size_t i = 0; i < pillar.point_num_in_pillar; i++) {
            size_t point_index = pillar.point_index[i];
            float curr_x = points[point_stride * point_index];
            float curr_y = points[point_stride * point_index + 1];
            float curr_z = points[point_stride * point_index + 2];
            mean_x += curr_x;
            mean_y += curr_y;
            mean_z += curr_z;

            // double check if current point is out-of-range point
            assert(curr_x >= MIN_X_RANGE && curr_x <= MAX_X_RANGE &&
                   curr_y >= MIN_Y_RANGE && curr_y <= MAX_Y_RANGE &&
                   curr_z >= MIN_Z_RANGE && curr_z <= MAX_Z_RANGE);
        }
        mean_x /= pillar.point_num_in_pillar;
        mean_y /= pillar.point_num_in_pillar;
        mean_z /= pillar.point_num_in_pillar;

        // calc centeral values of current pillar
        float x_center = (VOXEL_X_SIZE / 2.0f) +
                         (pillar.pillar_grid_x * VOXEL_X_SIZE) + MIN_X_RANGE;
        float y_center = (VOXEL_Y_SIZE / 2.0f) +
                         (pillar.pillar_grid_y * VOXEL_Y_SIZE) + MIN_Y_RANGE;
        float z_center = (VOXEL_Z_SIZE / 2.0f) + MIN_Z_RANGE;

        // write encoded features into Voxel
        for (size_t i = 0; i < pillar.point_num_in_pillar; i++) {
            size_t point_index = pillar.point_index[i];
            size_t voxel_index = i + MAX_NUM_POINTS_PER_PILLAR *
                                         (pillar.pillar_grid_x +
                                          pillar.pillar_grid_y * GRID_X_SIZE);
            voxels[voxel_index].x = points[point_stride * point_index];
            voxels[voxel_index].y = points[point_stride * point_index + 1];
            voxels[voxel_index].z = points[point_stride * point_index + 2];
            voxels[voxel_index].w = points[point_stride * point_index + 3];
            voxels[voxel_index].offset_from_mean_x =
                voxels[voxel_index].x - mean_x;
            voxels[voxel_index].offset_from_mean_y =
                voxels[voxel_index].y - mean_y;
            voxels[voxel_index].offset_from_mean_z =
                voxels[voxel_index].z - mean_z;

            voxels[voxel_index].offset_from_center_x =
                voxels[voxel_index].x - x_center;
            voxels[voxel_index].offset_from_center_y =
                voxels[voxel_index].y - y_center;
            voxels[voxel_index].offset_from_center_z =
                voxels[voxel_index].z - z_center;
        }
    }
}

void preprocess(const float *points, size_t points_buf_len,
                size_t point_stride) {
    std::vector<Pillar> bev_pillar(GRID_Y_SIZE * GRID_X_SIZE);
    std::vector<Voxel> voxels(GRID_Y_SIZE * GRID_X_SIZE *
                              MAX_NUM_POINTS_PER_PILLAR);
    voxelization(bev_pillar, points, points_buf_len, point_stride);
    point_decoration(bev_pillar, voxels, points, points_buf_len, point_stride);
}

} // namespace vueron