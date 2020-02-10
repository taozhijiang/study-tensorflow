#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

# 一些辅助进行mnist数据集处理的工具
from tensorflow.examples.tutorials.mnist import input_data

# 常数配置参数
INPUT_NODE_SIZE = 784 # 28*28
OUTPUT_NODE_SIZE = 10 # 0-9
LAYER1_NODE_SIZE = 512
LAYER2_NODE_SIZE = 256
LAYER3_NODE_SIZE = 128

INPUT_NODE_NAME  = "x-input"
OUTPUT_NODE_NAME = "y-output"
# softmax结果的最终预估结果
FINAL_OUTPUT_NODE_NAME = "final-output"
KEEP_PROB_NODE_NAME = "keep-prob"

# 训练的batch批次大小
BATCH_SIZE = 100

LEARN_RATE_BASE = 1e-4
TRAIN_STEPS = 1500
DROPOUT = 0.5  # 防治过拟合



def print_split():
    print("======================================================")


def freeze_graph(session, frozen_model_file):

    graph = tf.get_default_graph()
    #for op in graph.get_operations():
    #    print(" = save ops: ", op.name)

    # 通过Tensorflow的内置工具来导出variable
    # Tensorflow会根据依赖关系，知道需要保存哪些数据
    output_node_names = FINAL_OUTPUT_NODE_NAME
    output_graph_def = graph_util.convert_variables_to_constants(session, graph.as_graph_def(), output_node_names.split(","))

    with tf.gfile.GFile(frozen_model_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("[INFO] totally %s ops write to the final graph file %s." %(len(output_graph_def.node), frozen_model_file))
    return


def train(mnist, frozen_model_file):

    # 输入输出节点
    x  = tf.placeholder(tf.float32, shape=[None, INPUT_NODE_SIZE], name=INPUT_NODE_NAME)
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE_SIZE], name=OUTPUT_NODE_NAME)

    # 创建中间节点的权重和偏置量
    weights = {
        "w1":  tf.Variable(tf.truncated_normal([INPUT_NODE_SIZE,  LAYER1_NODE_SIZE], stddev=0.1)),
        "w2":  tf.Variable(tf.truncated_normal([LAYER1_NODE_SIZE, LAYER2_NODE_SIZE], stddev=0.1)),
        "w3":  tf.Variable(tf.truncated_normal([LAYER2_NODE_SIZE, LAYER3_NODE_SIZE], stddev=0.1)),
        "out": tf.Variable(tf.truncated_normal([LAYER3_NODE_SIZE, OUTPUT_NODE_SIZE], stddev=0.1)),
        }

    biases = {
        "b1":  tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE_SIZE])),
        "b2":  tf.Variable(tf.constant(0.1, shape=[LAYER2_NODE_SIZE])),
        "b3":  tf.Variable(tf.constant(0.1, shape=[LAYER3_NODE_SIZE])),
        "out": tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE_SIZE])),
        }

    # 定义神经网络结构
    # 各层节点的尺寸 784 - 512 - 256 - 128 - 10
    layer_1 = tf.add(tf.matmul(x, weights["w1"]), biases["b1"])
    layer_2 = tf.add(tf.matmul(layer_1, weights["w2"]), biases["b2"])
    layer_3 = tf.add(tf.matmul(layer_2, weights["w3"]), biases["b3"])

    # Dropout用来防治过度拟合的问题
    keep_prob = tf.placeholder(tf.float32, name=KEEP_PROB_NODE_NAME)
    layer_drop = tf.nn.dropout(layer_3, keep_prob)
    output_layer = tf.matmul(layer_drop, weights["out"]) + biases["out"]
    y_conv = tf.nn.softmax(output_layer, name=FINAL_OUTPUT_NODE_NAME)

    # 定义预测值和真实值之间的损失函数
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(lables=y_, logits=output_layer)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=output_layer)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 使用AdamOptimizer来优化
    train_step = tf.train.AdamOptimizer(LEARN_RATE_BASE).minimize(cross_entropy_mean)

    # 检查预估值和标签是否一致，用来计算预估准确性
    correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # 将boolean转成float，然后计算精度

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        # 验证数据集，通常用来验证训练的大致停止条件
        validate_feed = { x: mnist.validation.images,
                             y_: mnist.validation.labels }

        # 测试数据集，用来评判效果
        test_feed = { x: mnist.test.images,
                         y_: mnist.test.labels,
                         keep_prob: 1.0 }

        for step in range(TRAIN_STEPS):

            # 提取batch，用于随机梯度下降训练参数
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            session.run(train_step, feed_dict={x: xs, y_: ys, keep_prob: DROPOUT})

            # 因为数据集较小，可以进行全量测试计算准确率
            # 现实情况可以选取小的batch进行测试
            if step % 300 == 0:
                minibatch_accuracy = session.run(accuracy, feed_dict=test_feed)
                print("[INFO] Epoch %d with accuracy=%f."%(step, minibatch_accuracy))

        # 迭代结束，计算最终一轮的准确性
        test_accuracy = session.run(accuracy, feed_dict=test_feed)
        print("[INFO] final train accuracy=%f." %(test_accuracy))

        print("[INFO] save frozen_mode to file:", frozen_model_file)
        freeze_graph(session, frozen_model_file)

    return


def main(argv=None):

    mnist = input_data.read_data_sets("dataset/", one_hot=True)

    # 展示数据源信息
    print("[INFO] 训练数据源大小：", mnist.train.num_examples)
    print("[INFO] 验证数据源大小：", mnist.validation.num_examples)
    print("[INFO] 测试数据源大小：", mnist.test.num_examples)

    # 样例的数据打印
    #print("[INFO] ", mnist.train.images[0])

    print("[INFO] 测试数据源大小：", mnist.test.num_examples)
    print_split()

    frozen_model_file = "frozen_model.pb"
    train(mnist, frozen_model_file)

    return

if __name__ == '__main__':
    # TF框架自动调用上面的main方法
    tf.app.run()
