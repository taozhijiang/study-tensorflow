#include <iostream>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include  "tensorflow/cc/saved_model/tag_constants.h"

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "google/protobuf/map.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "apis/prediction_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

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

// server startup by:
// ./tensorflow_model_server --port=8500 --model_name="start_tf_demo" --model_base_path=/path/to...
// bin/start_tf_client /path/to...

const std::string str_input1_ops = "x-input1";
const std::string str_input2_ops = "x-input2";
const std::string str_output_ops = "y-output";

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

class ServingClient {
public:
    ServingClient(std::shared_ptr<Channel> channel)
        : stub_(PredictionService::NewStub(channel)) { }

    bool callPredict(const tensorflow::string& model_name,
                     const tensorflow::string& model_signature_name) {

        PredictRequest predictRequest;
        PredictResponse response;
        ClientContext context;

        predictRequest.mutable_model_spec()->set_name(model_name);
        predictRequest.mutable_model_spec()->set_signature_name(
            model_signature_name);

        google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs =
            *predictRequest.mutable_inputs();
        // 实例的请求数据
        tensorflow::TensorProto proto1;
        proto1.mutable_tensor_shape()->add_dim()->set_size(3);
        proto1.set_dtype(tensorflow::DataType::DT_INT32);
        for (size_t i = 0; i < 3; ++i) {
            proto1.add_float_val(static_cast<float>(i));
        }
        inputs[str_input1_ops] = proto1;

        tensorflow::TensorProto proto2;
        proto2.mutable_tensor_shape()->add_dim()->set_size(3);
        proto2.set_dtype(tensorflow::DataType::DT_INT32);
        for (size_t i = 0; i < 3; ++i) {
            proto2.add_float_val(static_cast<float>(i * 10));
        }
        inputs[str_input2_ops] = proto2;

        Status status = stub_->Predict(&context, predictRequest, &response);

        if (status.ok()) {

            //std::cout << "outputs size is " << response.outputs_size() << std::endl;
            google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& map_outputs = *response.mutable_outputs();

            for (auto iter = map_outputs.begin(); iter != map_outputs.end(); ++iter) {
                tensorflow::TensorProto& result_tensor_proto = iter->second;
                tensorflow::Tensor tensor;
                bool converted = tensor.FromProto(result_tensor_proto);
                if (converted) {
                    LOG(INFO) << "the result tensor is:" << tensor.DebugString();
                } else {
                   LOG(ERROR) << "the result tensor convert failed.";
                }
            }
            return  true;
        }


        LOG(ERROR) << "gRPC call return code: " << status.error_code() << ": " << status.error_message();
        return false;
    }

private:
    std::unique_ptr<PredictionService::Stub> stub_;
};

static void usage() {
    std::cout << "bin/start_tf_client [model_dir]" << std::endl;
    std::cout << "provide model_dir for more info." << std::endl;
}

int main(int argc, char* argv[]) {

    std::string saved_model_dir;
    if (argc >= 2)
        saved_model_dir = argv[1];

    if (!saved_model_dir.empty()) {

        tensorflow::SavedModelBundle bundle;
        tensorflow::SessionOptions session_options;
        tensorflow::RunOptions run_options;

        auto status = tensorflow::LoadSavedModel(session_options, run_options, saved_model_dir,
        { tensorflow::kSavedModelTagServe }, &bundle);
        if (!status.ok()) {
            LOG(ERROR) << "call LoadSavedModle failed at: " << saved_model_dir;
            LOG(ERROR) << status.error_message();
            return EXIT_FAILURE;
        }

        const auto signature_def_map = bundle.meta_graph_def.signature_def();
        const auto signature_def = signature_def_map.at("serving_default");

        auto inputs = signature_def.inputs();
        for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
            LOG(INFO) << "inputs: " << iter->first;
        }

        auto outputs = signature_def.outputs();
        for (auto iter = outputs.begin(); iter != outputs.end(); ++iter) {
            LOG(INFO) << "outputs: " << iter->first;
        }

        // TODO
        // bundle session inference

    }

    // gRPC
    tensorflow::string server_port = "localhost:8500";
    tensorflow::string model_name = "start_tf_demo";
    tensorflow::string model_signature_name = "serving_default";

    ServingClient guide(grpc::CreateChannel(server_port, grpc::InsecureChannelCredentials()));
    LOG(INFO) << "retcode: " << guide.callPredict(model_name, model_signature_name);

    return 0;
}
