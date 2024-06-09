#ifndef __RPN_H__
#define __RPN_H__

#include <vector>

/*
  Deprecated
*/

namespace vueron {

void rpn_run(const std::vector<float> &rpn_input,
             std::vector<std::vector<float>> &rpn_output);
}  // namespace vueron

#endif  // __RPN_H__
