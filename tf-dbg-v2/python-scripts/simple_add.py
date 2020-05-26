#!/usr/bin/env python3

import tensorflow as tf

# 逻辑
# ( x1 + x2 ) * weight + bias
# weight = 10, bias = 100


# saved_model_cli show --dir=./simple_add_v2/00001 --all

kInputNodeName1 = "x-input1"
kInputNodeName2 = "x-input2"
kOutputNodeName = "y-output"

class SimpleAddModel(tf.Module):

  def __init__(self):
    super(SimpleAddModel, self).__init__(name='SimpleModel')
    self.kWeight = tf.Variable(tf.constant(10), name="weight-n")
    self.kBias   = tf.Variable(tf.constant(100), name="bias-n")

  @tf.function(input_signature = [tf.TensorSpec(shape=None, dtype=tf.int32, name=kInputNodeName1),
                                  tf.TensorSpec(shape=None, dtype=tf.int32, name=kInputNodeName2)])
  def calc(self, x1, x2):
    return {kOutputNodeName:
            (x1 + x2) * self.kWeight + self.kBias + tf.constant(1)};



if __name__ == '__main__':
     
    model = SimpleAddModel()
    
    # 141 161
    # print(model.calc([1, 2], [3, 4]))

    export_dir = "./simple_add_v2/00001"
    print("[INFO] save model to ", export_dir)
    tf.saved_model.save(model, export_dir)


    loaded_model = tf.saved_model.load(export_dir)
    print("[INFO] ", repr(loaded_model))
    print("[INFO] ", repr(loaded_model.calc([1, 2], [3, 4])))
    print("[INFO] DONE.")

