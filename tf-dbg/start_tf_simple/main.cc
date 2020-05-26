#include <iostream>
#include <vector>

#include "add.hpp"

#include <glog/logging.h>

//
// * highlight comment
// TODO: todo tag
// ! notice comment tag
// ? query tag
//


// bin/start_tf_add ../../python-scripts/frozen_add_model.pb

int main(int argc, char *argv[]) {
  //
  // * Uncomment those if you want to use glog
  //
  // FLAGS_minloglevel = 0; // INFO
  // FLAGS_minloglevel = 1; // WARNING
  // FLAGS_minloglevel = 2; // ERROR
  // FLAGS_minloglevel = 3; // FATAL
  //
  // FLAGS_log_dir = "./logs";
  // google::InitGoogleLogging("start_tf_add");

  if (argc < 2) {
    add_usage();
    return EXIT_FAILURE;
  }

  const char *model_file = argv[1];
  LOG(INFO) << "using model file: " << model_file;

  tensorflow::Session *session = PrepareSession(model_file);
  if (!session) {
    LOG(ERROR) << "PrepareSession failed.";
    return EXIT_FAILURE;
  }

  if (!WarmUpSession(session)) {
    LOG(ERROR) << "WarmupSession failed.";
    return EXIT_FAILURE;
  }

  std::vector<int32_t> vec1{1, 3, 9};
  std::vector<int32_t> vec2{3, 5, 7};

  if (!RunInference(session, vec1, vec2)) {
    LOG(ERROR) << "RunInference failed.";
    return EXIT_FAILURE;
  }

  CloseSession(session);

  LOG(INFO) << "Tensorflow Inference finished.";
  return EXIT_SUCCESS;
}
