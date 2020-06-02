import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from Preprocess import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from saba_read import create_test_data,create_train_data
from DataSet import MapStyleDataset
import math

mypipline=Pipeline([('read_data',ReadData(data_dir='./data_new',file_name='cancer.csv')),
                    ('clean_data',CleanData('median'))
                    # ('cat_to_num',CatToNum('ordinal'))
                    # ,('Outlier_mitigation',OutlierDetection(threshold=2,name='whisker'))
                    ])
new_data= mypipline.fit_transform(None)
torch_writer= SummaryWriter('./Logs')
# data_train,data_test = train_test_split(new_data,test_size=.2,random_state=10)
#
# x_train_dense, x_train_sparse,x_train_init_seq,y_train_sequential= create_train_data(data_train)
# x_test_dense,x_test_sparse,x_test_init_seq,y_test_sequential= create_test_data(data_test)
class myLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        x, y = input.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = input.matmul(self.weight.t())
        if self.bias is not None:
            output += self.bias
        ret = output
        return ret

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

def myplot(y_true,y_pred,nrows=5,ncols=5,title='CRT1lymphocyte_absolute_count_KuL'):
    fig,axes= plt.subplots(nrows,ncols,sharex='none',sharey='none',figsize=(20,20))
    seq_len= y_true.shape[1]
    i=0
    for row_ax in axes:
        for col_ax in row_ax:
            col_ax.plot(np.arange(seq_len),y_true[i,:],color='darkblue',label='True_value')
            col_ax.plot(np.arange(seq_len),y_pred[i,:],color='darkorange',label='Pred_value')
            col_ax.legend()
            i+=1
    fig.savefig('./Figs/fig_{}.png'.format(title))

class SabaClass(nn.Module):
    def __init__(self,dense_size,sparse_size,
                 init_seq_size,hidd_size=20,seq_features=2,lstm_input_size=2,
                 lstm_hidd_size=2,lstm_layers=1,seq_size=1,sparse_penalty=.2,
                 max_seq=5,**kwargs):
        super(SabaClass,self).__init__(**kwargs)
        self.dense_size= dense_size
        self.sparse_size= sparse_size
        self.init_seq_size= init_seq_size
        self.hidd_size= hidd_size
        self.lstm_input_size= lstm_input_size
        self.lstm_hidd_size= lstm_hidd_size
        self.seq_features= seq_features
        self.lstm_layers= lstm_layers
        self.seq_size= seq_size
        self.max_seq= max_seq
        self.dense_layer=nn.Sequential(nn.Linear(in_features=self.dense_size,out_features=self.hidd_size,bias=True),
                                       nn.ReLU())
        self.sparse_layer= nn.Linear(in_features=self.sparse_size,out_features=self.hidd_size,bias=True)
        self.init_seq_layer=nn.Sequential(nn.Linear(in_features=self.init_seq_size,out_features=self.hidd_size,bias=True),
                                          nn.ReLU())

        self.combine_layer = nn.Sequential(nn.Linear(in_features=self.dense_size+self.sparse_size+self.init_seq_size,
                                      out_features=self.hidd_size),nn.ReLU())
        self.preprocess= nn.Sequential(nn.Linear(in_features=self.seq_features,out_features=self.hidd_size))

        self.lstm_layers_list=[
            nn.LSTM(input_size=self.hidd_size, hidd_size=self.seq_features,
                                      num_layers=self.lstm_layers) for _ in range(self.max_seq)
        ]

    def forward(self,x_dense,x_sparse,x_init_seq,hid_states):
        x_dense= self.dense_layer(x_dense)
        x_sparse = self.sparse_layer(x_sparse)
        x_init_seq= self.init_seq_layer(x_init_seq)
        x_all= self.combine_layer(torch.cat([x_dense,x_sparse,x_init_seq],dim=-1))
        x_seq = x_all.view(len(x_all),1,-1)
        list_outputs=[]
        for lstm in self.lstm_layers_list:
            init_state= hid_states
            x_seq,hid_states= lstm(x_seq,init_state)
            list_outputs.append(x_seq.view(-1,self.seq_features))
        return list_outputs

model= SabaClass()
optim= torch.optim.Adam(model.parameters(),lr=.01)
loss= torch.nn.MSELoss()
NUM_SEQ_Features=2

def train_step(x_dense,x_sparse,x_init_seq,init_state,y_seg_target):

    y_pred_list= model(x_dense,x_sparse,x_init_seq,init_state)
    step_loss=0
    for i,y_pred in enumerate(y_pred_list):
        y_targ = y_seg_target[:, i * NUM_SEQ_Features:(i + 1) * NUM_SEQ_Features]
        step_loss+= loss(y_targ,y_pred)

    optim.zero_grad()
    step_loss.backward()
    optim.step()
    return step_loss

mydataset= MapStyleDataset()
batch_size=32
loader = DataLoader(mydataset,batch_size= batch_size,shuffle=True)
training_mode= True
epochs=100

if training_mode== True:
    glob_step=0
    for epoch in range(epochs):
        for x_dense, x_sparse, x_init_seq, y_targ_seq in loader:
            curr_bacth= len(x_dense)
            init_states = (torch.zeros([1,curr_bacth,NUM_SEQ_Features]),
                           torch.zeros([1,curr_bacth,NUM_SEQ_Features]))

            batch_loss = train_step(x_dense,x_sparse,x_init_seq,init_states,y_targ_seq)
            torch_writer.add_scalar('train_loss',batch_loss,glob_step)
            if glob_step==0:
                torch_writer.add_graph(model,[x_dense, x_sparse, x_init_seq,init_states])
            if glob_step % 10:
                print('epoch {}/{}, iter {}, loss {}'.format(epoch,epochs,glob_step,
                                                             batch_loss.detach().numpy()))
            glob_step+=1

