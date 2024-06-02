#include "onnxruntime_cxx_api.h"
#include "params.h"
#include "type.h"
#include <cstddef>
#include <vector>

namespace vueron {
class PCDet {
    private:
    /*
        Buffers for Inference Pipeline
    */
    std::vector<Pillar> bev_pillar;
    std::vector<size_t> voxel_coords; // (x, y)
    std::vector<size_t> voxel_num_points;
    std::vector<float> pfe_input;  // input of pfe_run()
    std::vector<float> pfe_output; // input of scatter()
    std::vector<float> bev_image;  // input of RPN
    std::vector<std::vector<float>> rpn_outputs;

    /*
        Buffers for Final Predictions
    */
    std::vector<BndBox> pre_boxes;  // boxes before NMS
    std::vector<size_t> pre_labels; // labels before NMS
    std::vector<float> pre_scores;  // scores before NMS

    public:
    PCDet();
    ~PCDet();

    void preprocess();
    void pfe_run();
    void rpn_run();
    void postprocess();
    std::vector<PredBox> &get_pred();
};
} // namespace vueron
