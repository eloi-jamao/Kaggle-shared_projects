
import os
import numpy as np
import idx2numpy
import pandas as pd
import tensorflow as tf #version 2.0
from tensorflow import data
'''
# %% [code]

Kaggle paths to data files:
/kaggle/input/fashionmnist/train-labels-idx1-ubyte
/kaggle/input/fashionmnist/t10k-images-idx3-ubyte
/kaggle/input/fashionmnist/train-images-idx3-ubyte
/kaggle/input/fashionmnist/t10k-labels-idx1-ubyte
/kaggle/input/fashionmnist/fashion-mnist_test.csv
/kaggle/input/fashionmnist/fashion-mnist_train.csv

#load data on kaggle
train_dataset = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_dataset= pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

train_y = train_dataset.pop('label')
test_y = test_dataset.pop('label')

train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset.to_numpy(), train_y.to_numpy()))
test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset.to_numpy(), test_y.to_numpy()))
'''
logdir='/home/jamao/Desktop/programes/Kaggle_shared_projects/tensorboard_logs'

# %% [code]
#Local paths to data files:
test_X = '/home/jamao/Desktop/programes/data/Fashion MNIST/t10k-images-idx3-ubyte'
test_y = '/home/jamao/Desktop/programes/data/Fashion MNIST/t10k-labels-idx1-ubyte'
train_X = '/home/jamao/Desktop/programes/data/Fashion MNIST/train-images-idx3-ubyte'
train_y = '/home/jamao/Desktop/programes/data/Fashion MNIST/train-labels-idx1-ubyte'

#load data locally
train_X = idx2numpy.convert_from_file(train_X).reshape((60000,-1))
train_y = idx2numpy.convert_from_file(train_y)
test_X = idx2numpy.convert_from_file(test_X).reshape((10000,-1))
test_y = idx2numpy.convert_from_file(test_y)


train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))


# %% [code]
graph = tf.compat.v1.Graph()

with graph.as_default():
        
    with tf.compat.v1.variable_scope("Input"):
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset, train_y))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset, test_y))
        
        train = train_dataset.shuffle(10)
        train = train.batch(32)
        train = train.prefetch(10)
        
        test = test_dataset.shuffle(10)
        test = test.batch(32)
        test = test.prefetch(10)
    
        train_iter = train.make_one_shot_iterator()
        test_iter = test.make_one_shot_iterator()
        train_batch = train_iter.get_next()
        test_batch = test_iter.get_next()
        x, y = train_batch #to compute now, no placeholder or anything
        x = tf.reshape(x, [-1,28,28,1])
        
    with tf.compat.v1.variable_scope("FeedForward"):
        
        with tf.compat.v1.variable_scope("Conv_Bloc_1"):
            x = tf.keras.layers.Conv2d(filters = 6, kernel_size = (5,5), padding = (2,2), activation = 'relu', name="conv1")(x)
            x = tf.keras.layers.MaxPool2D(strides = (1,1), name="pool1")(x)
        
        with tf.compat.v1.variable_scope("Conv_Bloc_2"):
            x = tf.keras.layers.Conv2d(filters = 16, kernel_size = (5,5), activation = 'relu', name="conv1")(x)
            x = tf.keras.layers.MaxPool2D(strides = (1,1), name="pool1")(x)
        
        with tf.compat.v1.variable_scope("Dense_bloc"):
            x = tf.keras.layers.Flatten(data_format = 'channels_first')(x)
            x = tf.keras.layers.Dense(units=120, activation='relu')(x)
            x = tf.keras.layers.Dense(units=84, activation='relu')(x)
            y_ = tf.keras.layers.Dense(units=10, activation='softmax')(x)
    
sess = tf.compat.v1.Session()

with sess:
	sess.run(tf.compat.v1.global_variables_initializer())
	result = sess.run(y_)
	writer = tf.compat.v1.summary.FileWriter(logdir, graph=graph)
