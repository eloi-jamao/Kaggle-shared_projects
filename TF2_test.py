import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())
import tensorflow.compat.v1 as tf
print(tf.executing_eagerly())
tf.disable_eager_execution()
print(tf.executing_eagerly())