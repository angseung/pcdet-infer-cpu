#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>

struct IndexedValue {
    float value;
    size_t index;
};

std::vector<size_t> sort_and_get_indices(const std::vector<float> &vec) {
    std::vector<IndexedValue> indexed_values(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        indexed_values[i] = {vec[i], i};
    }

    std::sort(indexed_values.begin(), indexed_values.end(),
              [](const IndexedValue &a, const IndexedValue &b) {
                  return a.value < b.value;
              });

    std::vector<size_t> indices(vec.size());
    for (size_t i = 0; i < indexed_values.size(); ++i) {
        indices[i] = indexed_values[i].index;
    }
    assert(vec.size() == indices.size());

    return indices;
}
size_t find_index(const std::vector<float> &vec, float value) {
    auto it = std::find(vec.begin(), vec.end(), value);
    if (it != vec.end()) {
        return std::distance(vec.begin(), it); // 인덱스를 계산
    } else {
        return -1; // 값을 찾지 못한 경우 -1 반환
    }
}
