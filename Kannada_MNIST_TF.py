
import os
import numpy as np
import idx2numpy
import pandas as pd
import tensorflow as tf #version 2.1.0
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

'''
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
'''

# %% [code]
graph = tf.Graph()

with graph.as_default():
    
    with tf.variable_socope("Input"):
        train = train_dataset.shuffle(10).padded_batch(32, tf.TensorShape([32,32]), drop_remainder = True).prefetch(buffer_size)
        test = test_dataset.shuffle(10).padded_batch(32, tf.TensorShape([32,32]), drop_remainder = True).prefetch(buffer_size)
    
        train_iter = train.make_one_shot_iterator()
'''