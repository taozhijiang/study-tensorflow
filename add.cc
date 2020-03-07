#include <iostream>
#include <cassert>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>

const std::string str_input1_ops  = "x-input1:0";
const std::string str_input2_ops  = "x-input2:0";
const std::string str_output_ops = "y-output:0";

template <typename T>
tensorflow::Tensor AsTensor(const std::vector<T>& vals) {
    tensorflow::Tensor ret(tensorflow::DataTypeToEnum<T>::value, {static_cast<int>(vals.size())});
    std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
    return ret;
}

template <typename T>
tensorflow::Tensor AsTensor(const std::vector<T>& vals, const tensorflow::TensorShape& shape) {
    tensorflow::Tensor ret;
    ret.CopyFrom(AsTensor(vals), shape);
    return ret;
}


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

void CloseSession(tensorflow::Session* session) {
    if(session) {
        auto status = session->Close();
        if(!status.ok()) {
            std::cout << "[ERROR] Close Tensorflow session failed." << std::endl;
        } else {
            std::cout << "[INFO] Close Tensorflow session successfully." << std::endl;
        }

        delete session;
    }
}

// Tensorflow的计算是延迟加载的，所以为了避免模型在加载后首次预估超时，需要先做预热
bool WarmUpSession(tensorflow::Session* session) {
    if(!session)
        return false;

    tensorflow::Tensor input1(tensorflow::DT_INT32, tensorflow::TensorShape{1});
    input1.flat<int>().setConstant(1);
    tensorflow::Tensor input2(tensorflow::DT_INT32, tensorflow::TensorShape{1});
    input2.flat<int>().setConstant(1);

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {str_input1_ops, input1},
            {str_input2_ops, input2},
    };

    std::vector<tensorflow::Tensor> outputs {};
    tensorflow::Status status = session->Run(inputs, {str_output_ops}, {}, &outputs);
    if(!status.ok()) {
        std::cout << "[ERROR] Run session failed: " << status.error_message() << std::endl;
        return false;
    }

    if(outputs.size() != 1 ||
       outputs[0].dims() != 1 ||
       outputs[0].dim_size(0) != 1 ) {
           std::cout << "[ERROR] Warmup failed: " << std::endl;
           return false;
    }

    // value: (1 + 1) * 10 + 100 == 120
    std::cout << " warmup value:" << outputs[0].flat<int>()(0) << std::endl;
    std::cout << "[INFO] End Tensorflow session warmup ..." << std::endl;
    return true;
}

bool RunInference(tensorflow::Session* session, const std::vector<int>& vec1, const std::vector<int>& vec2) {
    if(!session)
        return false;

    tensorflow::Tensor input1 = AsTensor(vec1);
    tensorflow::Tensor input2 = AsTensor(vec2);
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {str_input1_ops, input1},
            {str_input2_ops, input2},
    };

    std::vector<tensorflow::Tensor> outputs {};
    tensorflow::Status status = session->Run(inputs, {str_output_ops}, {}, &outputs);
    if(!status.ok()) {
        std::cout << "[ERROR] Run session failed: " << status.error_message() << std::endl;
        return false;
    }

    if(outputs.size() != 1 ||
       outputs[0].dims() != 1 ||
       outputs[0].dim_size(0) != vec1.size() ) {
           std::cout << "[ERROR] Warmup failed: " << std::endl;
           return false;
    }

    auto out_flat = outputs[0].flat<int>();
    for(size_t i=0; i<vec1.size(); ++i) {
        std::cout << "result<" << i << "> :" << out_flat(i) << std::endl;
    }
  
    return true;
}

static void usage() {
    std::cout << "simple tensorflow exec demo:" << std::endl;
    std::cout << "bin/start_add path/to/frozen-model.pb " << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {

    if(argc < 2) {
        usage();
        return EXIT_FAILURE;
    }

    const char* model_file = argv[1];
    std::cout << "using model file: " << model_file << std::endl;

    tensorflow::Session* session = PrepareSession(model_file);
    if(!session) {
        std::cout << "[ERROR] PrepareSession failed." << std::endl;
        return EXIT_FAILURE;
    }

    if(!WarmUpSession(session)) {
        std::cout << "[ERROR] WarmupSession failed." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<int32_t> vec1{1, 3, 9};
    std::vector<int32_t> vec2{3, 5, 7};

    if(!RunInference(session, vec1, vec2)) {
        std::cout << "[ERROR] RunInference failed." << std::endl;
        return EXIT_FAILURE;
    }

    CloseSession(session);

    std::cout << "[INFO] Tensorflow Inference finished." << std::endl;
    return EXIT_SUCCESS;
}
