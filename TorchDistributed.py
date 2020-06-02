import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

class LinearLayer(torch.nn.Module):
    def __init__(self,input_features,output_features,**kwargs):
        super(LinearLayer,self).__init__(**kwargs)
        self.output_features=output_features
        self.input_features= input_features
        self.weight= nn.Parameter(torch.randn(size=[self.output_features,self.input_features],
                                              requires_grad=True,dtype=torch.float32))
        self.bias= nn.Parameter(torch.randn(size=[self.output_features],
                                              requires_grad=True,dtype=torch.float32))

    def forward(self, x):
        return torch.add(torch.matmul(x,self.weight.t()),self.bias)

in_= torch.randn(size=[5,2],requires_grad=False)
mylayer= LinearLayer(2,1)
y_true= torch.randn(size=[5,1],requires_grad=False)
y_pred= mylayer(in_)
optim= torch.optim.Adam(mylayer.parameters(),lr=.2)
loss= torch.mean(torch.sum((y_pred-y_true)**2))
optim.zero_grad()
loss.backward()
optim.step()
