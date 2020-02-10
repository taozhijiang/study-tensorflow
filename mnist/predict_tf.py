#!/usr/bin/env python3

import tensorflow as tf

# 图像处理的工具
import numpy as np
# package Pillow
from PIL import Image

INPUT_NODE_NAME = "x-input:0"
FINAL_OUTPUT_NODE_NAME = "final-output:0"
KEEP_PROB_NODE_NAME = "keep-prob:0"


def load_graph(frozen_model_file):

    # 加载pb格式的模型
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(frozen_model_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    #for op in graph.get_operations():
    #    print(" = load ops:", op.name)
    
    return graph


if __name__ == '__main__':

    frozen_model_file = "frozen_model.pb"
    graph = load_graph(frozen_model_file)

    t_input = graph.get_tensor_by_name(INPUT_NODE_NAME)
    t_output = graph.get_tensor_by_name(FINAL_OUTPUT_NODE_NAME)
    t_keep_prob = graph.get_tensor_by_name(KEEP_PROB_NODE_NAME)

    print("[INFO] t_input: ",  repr(t_input))
    print("[INFO] t_output: ", repr(t_output))

    # 识别预估
    # current matrix represents black as 0 and white as 255, whereas we need the opposite
    img = np.invert(Image.open("test_img.png").convert('L')).ravel()
    print("img size:", len(img))

    with tf.Session(graph=graph) as session:
        prediction = session.run(tf.argmax(t_output, 1), feed_dict={t_input: [img], t_keep_prob: 1.0})
        # np.squeeze function is called to return the single integer
        print("t_output:", np.squeeze(prediction))
