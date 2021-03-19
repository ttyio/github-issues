import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import graph_util
x = tf.placeholder(tf.float32, [1, 2, 4, 1], name="input")
y = tf.image.resize_bilinear(x, size=[1, 2], align_corners=True, half_pixel_centers=False)
_ = tf.identity(y, name="output")
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
    with tf.gfile.FastGFile('tfmodel.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    os.system("python3 -m tf2onnx.convert --opset 11 --input tfmodel.pb --inputs input:0 --outputs output:0 --output tfmodel.onnx")
