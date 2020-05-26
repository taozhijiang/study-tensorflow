#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

def freeze_graph(session, outputnodes, frozen_model_file):

    graph = tf.get_default_graph()

    # 通过Tensorflow的内置工具来导出variable
    # Tensorflow会根据依赖关系，知道需要保存哪些数据
    output_graph_def = graph_util.convert_variables_to_constants(session, graph.as_graph_def(), outputnodes.split(","))

    with tf.gfile.GFile(frozen_model_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("[INFO] totally %s ops write to the final graph file %s." %(len(output_graph_def.node), frozen_model_file))
    return
