#!/usr/bin/env python3

import model_util
import tensorflow as tf

# saved model
import tensorflow.python.saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

kInputNodeSize = 1
kInputNodeName1 = "x-input1"
kInputNodeName2 = "x-input2"
kOutputNodeSize = 1
kOutputNodeName = "y-output"


# 输入输出节点
x1 = tf.placeholder(tf.int32, shape=[None, kInputNodeSize], name=kInputNodeName1)
x2 = tf.placeholder(tf.int32, shape=[None, kInputNodeSize], name=kInputNodeName2)

kWeight = tf.Variable(tf.constant(10), name="weight-n")
with tf.colocate_with(kWeight):
    kBias   = tf.Variable(tf.constant(100), name="bias-n")


def run_demo(records1, records2):

    op_sum = tf.add(x1, x2)
    op_mul = tf.multiply(op_sum, kWeight)
    with tf.device('/device:CPU:0'):
        y = tf.add(op_mul, kBias, name=kOutputNodeName)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        result = session.run(y, feed_dict = {x1 : records1, x2: records2})
        print(result)

        frozen_model_file = "frozen_add_model.pb"
        print("[INFO]frozen the model to ", frozen_model_file)
        model_util.freeze_graph(session, kOutputNodeName, frozen_model_file)

        # SavedModel
        export_path = "./saved_simple_add/00001"
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        signature = predict_signature_def(
                inputs = {
                kInputNodeName1: x1,
                kInputNodeName2: x2,
                },
                outputs = {kOutputNodeName: y})
        builder.add_meta_graph_and_variables(sess=session,
                                     tags=[tf.saved_model.tag_constants.SERVING],
                                     signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})

        builder.save()
        print("[INFO] savedmodel to ", export_path)

def main(argv=None):

    # define feed operands
    # (x0 + x1)  * 10 + 100
    input_x1 = [ [2], [3], [4], [5]]
    input_x2 = [ [0], [2], [3], [1]]
    run_demo(input_x1, input_x2)

if __name__ == '__main__':
    tf.app.run()