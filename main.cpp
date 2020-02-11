#include <iostream>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>

tensorflow::Session* PrepareSession(const std::string& model_file) {

    // 加载预估模型
    tensorflow::GraphDef graph_def {};
    tensorflow::Session* session = NULL;

    auto status = ReadBinaryProto(tensorflow::Env::Default(), model_file, &graph_def);
    if(!status.ok()) {
        std::cout << "[ERROR] read model_file from " << model_file << "failed: " << status.ToString() << std::endl;
        return nullptr;
    }

    // 创建Session，各种选项可以选择
    auto session_option = tensorflow::SessionOptions();
    session_option.config.set_intra_op_parallelism_threads(10);
    session_option.config.set_inter_op_parallelism_threads(2);
    session_option.config.set_use_per_session_threads(false);

    status = NewSession(session_option, &session);
    if(!status.ok()) {
        std::cout << "[ERROR] new session failed: " << status.ToString() << std::endl;
        return nullptr;
    }

    status = session->Create(graph_def);
    if(!status.ok()) {
        std::cout << "[ERROR] session create with graph failed: " << status.ToString() << std::endl;
        return nullptr;
    }

    return session;
}

// Tensorflow的计算是延迟加载的，所以为了避免模型在加载后首次预估超时，需要先做预热
bool WarmUpSession(tensorflow::Session* session) {
    if(!session)
        return false;

    std::cout << "[INFO] Begin Tensorflow session warmup ..." << std::endl;

    // TODO ..

    std::cout << "[INFO] End Tensorflow session warmup ..." << std::endl;
    return true;
}

bool RunInference(tensorflow::Session* session) {
    return true;
}

int main(int argc, char* argv[]) {

    const char* model_file = "../mnist/frozen_model.pb";
    tensorflow::Session* session = PrepareSession(model_file);
    if(!session) {
        std::cout << "[ERROR] PrepareSession failed." << std::endl;
        return EXIT_FAILURE;
    }


    std::cout << "[INFO] Tensorflow Inference finished." << std::endl;
    return EXIT_SUCCESS;
}
