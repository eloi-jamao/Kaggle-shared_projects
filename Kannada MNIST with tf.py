# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
#importing
import numpy as np 
import pandas as pd 
import tensorflow as tf #version 2.1.0

# %% [code]
#loading datasets
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

# %% [code]
y_train = train.pop("label")
X_train = train
X_test = test.drop("id", axis = 1)
'''
print(y_train.head())
print(X_train.head())
print(X_test.head())
'''
train = tf.data.Dataset.from_tensor_slices((X_train.to_numpy(), y_train.to_numpy()))




# %% [code]
#Creating the graph
graph = tf.Graph()
with graph.as_default():
    train_dataset = train.shuffle(len(train)).batch(1)