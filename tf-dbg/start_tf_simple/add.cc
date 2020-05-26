#include "add.hpp"
#include "tensor_util.hpp"

#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/version.h>

#undef CHECK
#undef CHECK_GT
#undef CHECK_GE
#undef CHECK_NOTNULL
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_LT
#undef CHECK_EQ
#undef CHECK_OP
#undef CHECK_OP_LOG
#undef LOG
#undef VLOG
#undef VLOG_IS_ON

#include <glog/logging.h>

tensorflow::Session *PrepareSession(const std::string &model_file) {
  // 加载预估模型
  tensorflow::GraphDef graph_def{};
  tensorflow::Session *session = NULL;

  auto status =
      ReadBinaryProto(tensorflow::Env::Default(), model_file, &graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "read model_file from " << model_file
               << "failed: " << status.ToString();
    return nullptr;
  }

  // 创建Session，各种选项可以选择
  auto session_option = tensorflow::SessionOptions();
  status = NewSession(session_option, &session);
  if (!status.ok()) {
    LOG(ERROR) << "new session failed: " << status.ToString();
    return nullptr;
  }

  status = session->Create(graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "session create with graph failed: " << status.ToString();
    return nullptr;
  }

  return session;
}

void CloseSession(tensorflow::Session *session) {
  if (session) {
    auto status = session->Close();
    if (!status.ok()) {
      LOG(ERROR) << "close Tensorflow session failed.";
    } else {
      LOG(INFO) << "close Tensorflow session successfully.";
    }

    delete session;
  }
}

// Tensorflow的计算是延迟加载的，所以为了避免模型在加载后首次预估超时，需要先做预热
bool WarmUpSession(tensorflow::Session *session) {
  if (!session) {
    LOG(ERROR) << "session null.";
    return false;
  }

  tensorflow::Tensor input1(tensorflow::DT_INT32, tensorflow::TensorShape{1});
  input1.flat<int>().setConstant(1);
  tensorflow::Tensor input2(tensorflow::DT_INT32, tensorflow::TensorShape{1});
  input2.flat<int>().setConstant(1);

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {str_input1_ops, input1},
      {str_input2_ops, input2},
  };

  std::vector<tensorflow::Tensor> outputs{};
  tensorflow::Status status =
      session->Run(inputs, {str_output_ops}, {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Run session failed: " << status.error_message();
    return false;
  }

  if (outputs.size() != 1 || outputs[0].dims() != 1 ||
      outputs[0].dim_size(0) != 1) {
    LOG(ERROR) << "Warmup failed: ";
    return false;
  }

  // value: (1 + 1) * 10 + 100 == 120
  LOG(INFO) << "warmup value:" << outputs[0].flat<int>()(0);
  LOG(INFO) << "End Tensorflow session warmup ...";
  return true;
}

bool RunInference(tensorflow::Session *session, const std::vector<int> &vec1,
                  const std::vector<int> &vec2) {
  if (!session) {
    LOG(ERROR) << "session null.";
    return false;
  }

  tensorflow::Tensor input1 = AsTensor(vec1);
  tensorflow::Tensor input2 = AsTensor(vec2);
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {str_input1_ops, input1},
      {str_input2_ops, input2},
  };

  std::vector<tensorflow::Tensor> outputs{};
  tensorflow::Status status =
      session->Run(inputs, {str_output_ops}, {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Run session failed: " << status.error_message();
    return false;
  }

  if (outputs.size() != 1 || outputs[0].dims() != 1 ||
      outputs[0].dim_size(0) != vec1.size()) {
    LOG(ERROR) << "Warmup failed: ";
    return false;
  }

  auto out_flat = outputs[0].flat<int>();
  for (size_t i = 0; i < vec1.size(); ++i) {
    LOG(INFO) << "result<" << i << "> :" << out_flat(i);
  }

  return true;
}

void add_usage() {
  std::cout << "simple tensorflow exec demo:" << std::endl;
  std::cout << "bin/start_tf_add path/to/frozen-model.pb " << std::endl;
  std::cout << std::endl;
}
