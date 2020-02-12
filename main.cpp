#include <iostream>
#include <cassert>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>

#include "img_data.inc"

const std::string str_input_ops  = "x-input:0";
const std::string str_output_ops = "final-output:0";
const std::string str_keep_prob = "keep-prob:0";
const size_t kInputSize = 784;


tensorflow::Tensor MakeMnistTensor(const float images[]) {

    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape{1, kInputSize});
    assert(tensor.NumElements() == kInputSize);
    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        tensor.flat<float>()(i) = images[i];
    }

    return tensor;
}

tensorflow::Tensor MakeMnistTensor(const float images_a[], const float images_b[]) {

    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape{2, kInputSize});
    assert(tensor.NumElements() == 2 * kInputSize);
    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        if(i < kInputSize)
            tensor.flat<float>()(i) = images_a[i];
        else
            tensor.flat<float>()(i) = images_b[i - kInputSize];
    }

    return tensor;
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
bool WarmUpSession(tensorflow::Session* session, float images[]) {
    if(!session)
        return false;

    tensorflow::Tensor tens = MakeMnistTensor(images);
    tensorflow::Tensor keep_prob(tensorflow::DT_FLOAT, tensorflow::TensorShape{1});
    keep_prob.flat<float>().setConstant(1.0);


    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {str_input_ops, tens},
            {str_keep_prob, keep_prob},
    };

    std::vector<tensorflow::Tensor> outputs {};

    tensorflow::Status status = session->Run(inputs, {str_output_ops}, {}, &outputs);
    if(!status.ok()) {
        std::cout << "[ERROR] Run session failed: " << status.error_message() << std::endl;
        return false;
    }

    std::cout << "[INFO] End Tensorflow session warmup ..." << std::endl;
    return true;
}


struct prob_result {
    bool valid_ = false;
    int data_;      // 数值
    float prob_;    // 置信度
};

struct prob_result MakeResult(const float* predict_out) {
    
    struct prob_result result {};

    if(!predict_out)
        return result;

    for(size_t i=0; i<10; ++i) {
        if(*(predict_out + i) > result.prob_) {
            result.data_ = i;
            result.prob_ = *(predict_out + i);
        }
    }

    result.valid_ = true;
    return result;
}

bool RunInference(tensorflow::Session* session, const float images_a[], const float images_b[]) {
    if(!session)
        return false;

    tensorflow::Tensor tens = MakeMnistTensor(images_a, images_b);
    tensorflow::Tensor keep_prob(tensorflow::DT_FLOAT, tensorflow::TensorShape{1});
    keep_prob.flat<float>().setConstant(1.0);


    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {str_input_ops, tens},
            {str_keep_prob, keep_prob},
    };

    std::vector<tensorflow::Tensor> outputs {};

    tensorflow::Status status = session->Run(inputs, {str_output_ops}, {}, &outputs);
    if(!status.ok()) {
        std::cout << "[ERROR] Run session failed: " << status.error_message() << std::endl;
        return false;
    }

    const auto result = outputs[0];
    // std::cout << "[INFO] result: " << outputs[0].DebugString() << std::endl;
    assert(result.shape().dim_size(0) == 2);
    assert(result.shape().dim_size(1) == 10);

    const auto data = result.flat<float>().data();
    auto result1 = MakeResult(data);
    auto result2 = MakeResult(data + 10);

    std::cout << "result1 => data: " << result1.data_ << ", prob: " << result1.prob_ << std::endl;
    std::cout << "result2 => data: " << result2.data_ << ", prob: " << result2.prob_ << std::endl;

    return true;
}

int main(int argc, char* argv[]) {

    const char* model_file = "../mnist/frozen_model.pb";
    // std::cout << "size of image_a: " << sizeof(image_a) / sizeof(image_a[0]) << std::endl;
    // std::cout << "size of image_b: " << sizeof(image_b) / sizeof(image_b[0]) << std::endl;

    tensorflow::Session* session = PrepareSession(model_file);
    if(!session) {
        std::cout << "[ERROR] PrepareSession failed." << std::endl;
        return EXIT_FAILURE;
    }

    if(!WarmUpSession(session, image_a)) {
        std::cout << "[ERROR] WarmupSession failed." << std::endl;
        return EXIT_FAILURE;
    }

    if(!RunInference(session, image_a, image_b)) {
        std::cout << "[ERROR] RunInference failed." << std::endl;
        return EXIT_FAILURE;
    }


    std::cout << "[INFO] Tensorflow Inference finished." << std::endl;
    return EXIT_SUCCESS;
}
