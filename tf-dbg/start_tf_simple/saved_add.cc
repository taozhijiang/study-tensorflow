#include <iostream>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

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

#include "tensor_util.hpp"

#include <glog/logging.h>

const std::string str_input1_ops = "x-input1";
const std::string str_input2_ops = "x-input2";
const std::string str_output_ops = "y-output";

// bin/start_tf_saved_add ../../python-scripts/saved_simple_add/00001
static void usage() {
  std::cout << "bin/start_tf_saved_add path/to/saved_model" << std::endl;
}

int main(int argc, char* argv[]) {
  std::string model_dir;
  if (argc < 2) {
    usage();
    return EXIT_FAILURE;
  }

  const std::string saved_model_dir = argv[1];

  tensorflow::SavedModelBundle bundle;
  tensorflow::SessionOptions session_options;
  tensorflow::RunOptions run_options;

  auto status =
      tensorflow::LoadSavedModel(session_options, run_options, saved_model_dir,
                                 {tensorflow::kSavedModelTagServe}, &bundle);
  if (!status.ok()) {
    LOG(ERROR) << "call LoadSavedModle failed at: " << saved_model_dir
               << std::endl;
    LOG(ERROR) << status.error_message() << std::endl;
    return EXIT_FAILURE;
  }

  const auto signature_def_map = bundle.meta_graph_def.signature_def();
  const auto signature_def = signature_def_map.at("serving_default");

  auto inputs_def = signature_def.inputs();
  for (auto iter = inputs_def.begin(); iter != inputs_def.end(); ++iter) {
    LOG(INFO) << "inputs_def: " << iter->first << std::endl;
  }

  auto outputs_def = signature_def.outputs();
  for (auto iter = outputs_def.begin(); iter != outputs_def.end(); ++iter) {
    LOG(INFO) << "outputs_def: " << iter->first << std::endl;
  }

  std::vector<int32_t> vec1{1, 3, 9};
  std::vector<int32_t> vec2{3, 5, 7};

  tensorflow::Tensor input1 = AsTensor(vec1);
  tensorflow::Tensor input2 = AsTensor(vec2);
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {str_input1_ops, input1},
      {str_input2_ops, input2},
  };

  std::vector<tensorflow::Tensor> outputs{};
  status = bundle.session->Run(inputs, {str_output_ops}, {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Run session failed: " << status.error_message();
    return EXIT_FAILURE;
  }

  if (outputs.size() != 1 || outputs[0].dims() != 1 ||
      outputs[0].dim_size(0) != vec1.size()) {
    LOG(ERROR) << "Warmup failed: ";
    return EXIT_FAILURE;
  }

  auto out_flat = outputs[0].flat<int>();
  for (size_t i = 0; i < vec1.size(); ++i) {
    LOG(INFO) << "result<" << i << "> :" << out_flat(i);
  }

  return EXIT_SUCCESS;
}
