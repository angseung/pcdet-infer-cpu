#ifndef __RPN_H_TEST__
#define __RPN_H_TEST__

#include <vector>

namespace vueron {

void rpn_run(const std::vector<float> &rpn_input,
             std::vector<std::vector<float>> &rpn_output);
}  // namespace vueron

#endif  // __RPN_H_TEST__
