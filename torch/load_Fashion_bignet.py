import torch
import torch.nn as nn



class ConvBlock(nn.Module):

  def __init__(self, num_inp_channels, num_out_fmaps, 
               kernel_size, stride=1):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.conv = nn.Conv2d(num_inp_channels, num_out_fmaps, kernel_size,
                          stride=stride)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    P_ = self.kernel_size // 2
    if self.stride > 1:
      P = (P_ - 1, P_, P_  - 1, P_ )
    else:
      P = (P_, P_, P_, P_)
    x = self.conv(F.pad(x, P, mode='constant'))
    return self.relu(x)

class BigNetDropout(nn.Module):

  def __init__(self, dropout):
    super().__init__()
    self.conv1 = ConvBlock(1, 10, 3, stride=2)
    self.drop1 = nn.Dropout(dropout)
    self.conv2 = ConvBlock(10, 32, 3, stride=2)
    self.drop2 = nn.Dropout(dropout)
    self.mlp = nn.Sequential(
        nn.Linear(32*7*7, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(128, hparams['num_classes']),
        nn.LogSoftmax(dim=-1)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.drop1(x)
    x = self.conv2(x)
    x = self.drop2(x)
    bsz, nch, height, width = x.shape
    #print(x.shape)
    x = x.view(bsz, -1)
    y = self.mlp(x)
    return y

if __name__ == '__main__':

	state_dict = '' #add path to saved state_dict of the trained model

	bignet = BigNetDropout(dropout = 0)
	bignet.load_state_dict(torch.load(state_dict))

	#Here the model is ready to make predictions
	