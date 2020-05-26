#ifndef __START_TF_SIMPLE_ADD__
#define __START_TF_SIMPLE_ADD__

#include <string>
namespace tensorflow {
class Session;
}

const std::string str_input1_ops = "x-input1:0";
const std::string str_input2_ops = "x-input2:0";
const std::string str_output_ops = "y-output:0";

tensorflow::Session *PrepareSession(const std::string &model_file);
void CloseSession(tensorflow::Session *session);

// Tensorflow的计算是延迟加载的，所以为了避免模型在加载后首次预估超时，需要先做预热
bool WarmUpSession(tensorflow::Session *session);
bool RunInference(tensorflow::Session *session, const std::vector<int> &vec1,
                  const std::vector<int> &vec2);

void add_usage();

#endif  // __START_TF_SIMPLE_ADD__