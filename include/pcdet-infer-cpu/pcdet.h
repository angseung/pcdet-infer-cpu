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
    size_t num_pillars;            // for scatter
    std::vector<float> pfe_input;  // input of pfe_run()
    std::vector<float> pfe_output; // input of scatter()
    std::vector<float> bev_image;  // input of RPN
    std::vector<std::vector<float>> rpn_outputs;
    std::vector<BndBox> pre_boxes;  // boxes before NMS
    std::vector<size_t> pre_labels; // labels before NMS
    std::vector<float> pre_scores;  // scores before NMS
    std::vector<bool> suppressed;   // mask for nms

    /*
        Buffers for Final Predictions
    */
    std::vector<BndBox> post_boxes;  // boxes after NMS
    std::vector<size_t> post_labels; // labels after NMS
    std::vector<float> post_scores;  // scores after NMS

    public:
    PCDet(void);
    ~PCDet(void);

    void preprocess(const float *points, const size_t point_buf_len,
                    const size_t point_stride);
    void pfe_run(void);
    void scatter(void);
    void rpn_run(void);
    void postprocess(std::vector<vueron::BndBox> &post_boxes,
                     std::vector<size_t> &post_labels,
                     std::vector<float> &post_scores);
    void get_pred(std::vector<PredBox> &boxes);
    void do_infer(const float *points, const size_t point_buf_len,
                  const size_t point_stride, std::vector<PredBox> &boxes);
    void do_infer(const float *points, const size_t point_buf_len,
                  const size_t point_stride,
                  std::vector<vueron::BndBox> &final_boxes,
                  std::vector<size_t> &final_labels,
                  std::vector<float> &final_scores);
};
} // namespace vueron
