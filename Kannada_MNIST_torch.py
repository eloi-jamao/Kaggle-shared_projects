# %% [code]
import numpy as np 
import torch 
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
import itertools
from torchvision import transforms, utils
import torch.utils.data as data_utils
import statistics
import matplotlib.pyplot as plt
%matplotlib inline 
import torch.optim as optim


# %% [code]
params = {
    'batch_size':32,
    'num_epochs':10,
    'hidden_size':128,
    'num_classes':10,
    'num_inputs':784,
    'learning_rate':1e-3,
    }

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#print(device)


# %% [code]
#Import datasets
dataset = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
validation_dataset = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv") 

# %% [code]
#Separate labels and vectors an one hot encoding
y_dataset, X_dataset = dataset["label"], dataset.drop("label", axis=1)
y_dataset, X_dataset = torch.tensor(y_dataset.to_numpy(), dtype = torch.long), torch.tensor(X_dataset.to_numpy(), dtype = torch.float32)/255.

#changing to a torch.dataset
train = data_utils.TensorDataset(X_dataset, y_dataset)
train_dataset, test_dataset, trash = torch.utils.data.random_split(train, [55000, 5000,0])
train_loader = data_utils.DataLoader(train_dataset, batch_size = params["batch_size"], shuffle = True)
test_loader = data_utils.DataLoader(test_dataset, batch_size = params["batch_size"], shuffle = True)

#for i in range(len(X_dataset[0])):
#    X_dataset[i,:] = (X_dataset[i,:] - statistics.mean(X_dataset[i,:]))/statistics.sdev(X_dataset[i,:])
    

# %% [code]
#Definition of the class to hold the net arquitecture
class FullyConnected(nn.Module):
  
  def __init__(self, num_inputs, hidden_size, num_classes):
    super().__init__()
    
    self.h1 = nn.Linear(num_inputs,hidden_size)
    self.a1 = nn.ReLU()
    self.h2 = nn.Linear(hidden_size,num_classes)
    self.a2 = nn.LogSoftmax(dim = 1)
    
  def forward(self, x):
    y = self.a2(self.h2(self.a1(self.h1(x))))
    
    return y

#Function to get the number of parameters in the network
def get_nn_nparams(net):
  pp=0
  for p in list(net.parameters()):
      nn=1
      for s in list(p.size()):
          nn = nn*s
      pp += nn
  return pp


MNet = FullyConnected(num_inputs = params['num_inputs'], hidden_size = params['hidden_size'] ,num_classes = params['num_classes'])

loss = F.cross_entropy

opt = optim.SGD(MNet.parameters(), lr = params['learning_rate'], momentum = 0.9)



# %% [code]
##ENTENDRE LA FUNCIÓ CORRECT_PREDICTIONS

def correct_predictions(predicted_batch, label_batch):
  pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
  acum = pred.eq(label_batch.view_as(pred)).sum().item()
  return acum


def train_epoch(train_loader, network, optimizer, loss_func, params, loss_array):
    network.train()
    avg_loss = 0
    for idx, (data, label) in enumerate(train_loader):
        data, y = data.to(device), label.to(device)
        optimizer.zero_grad()
        #label = label.view(label.shape[0], -1)
        
        y_ = network(data)
        loss_val = loss_func(y_, label)
        loss_array.append(loss_val.item())
        #print(idx, loss_val.item())
        avg_loss += (loss_val)
        loss_val.backward(torch.ones(params['num_classes'], dtype = torch.float))
        optimizer.step()
        
    avg_loss /= float(len(train_loader.dataset))   
    
    #return avg_loss

def test_epoch(test_loader, network, loss_func, params, loss_array):
    acc = 0
    avg_loss = 0
    for data, label in test_loader:
        data, y = data.to(device), label.to(device)
        y_ = network.forward(data)

        loss_val = loss_func(y_, label)
        loss_array.append(loss_val.item())
        avg_loss += loss_val
        
        acc += correct_predictions(y_, label)

    avg_loss /= float(len(test_loader.dataset))
    test_acc = 100. * acc / len(test_loader.dataset)
    
    return avg_loss, test_acc

# %% [code]
train_losses = []
test_losses = []
test_acc = []


for epoch in range(params["num_epochs"]):

    print(epoch)
    train_epoch(train_loader, MNet, opt, loss, params, train_losses)
    epoch_loss, epoch_acc = test_epoch(test_loader, MNet, loss, params, test_losses)
    test_acc.append(epoch_acc)
    #test_losses.append(epoch_loss)
    #test_acc.append(epoch_acc)
    #print(train_losses)
    #print(test_losses)

figure, (train_plot, test_plot, acc_plot) = plt.subplots(3, figsize = (10,15)) 
train_plot.plot(train_losses)
train_plot.set_title("Train Dataset Loss")
test_plot.plot(test_losses)
test_plot.set_title("Test Dataset Loss")
acc_plot.plot(test_acc)
acc_plot.set_title("Test Dataset Accuracy")
plt.xlabel('Epochs')
plt.show()
    
    