import numpy as np
import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())
import tensorflow.compat.v1 as tf
print(tf.executing_eagerly())
tf.disable_eager_execution()
print(tf.executing_eagerly())

logdir='tensorboard_logs/test'

graph = tf.Graph()

with graph.as_default():

	X = tf.random_normal(shape = [2,2], dtype = tf.float32, name = 'input')
	y = 3 * X

sess = tf.Session(graph = graph)

with sess:
	sess.run(tf.global_variables_initializer())
	result = sess.run(y)

writer = tf.summary.FileWriter(logdir, graph = graph)

print(result)