import math
import numpy as np

import torch
from torch import nn as nn

import sys
sys.path.append("..")

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, 
                dataframe, 
                isValSet_bool=None, 
                sequence_length=30,
                test_size=0.2,
                ):

        len_test = len(dataframe) - int(len(dataframe) * test_size)

        self.dataframe = dataframe
        self.target= "Close"
        self.features = ['High', 'Low', 'Open', 'Volume']

        self.sequence_length = sequence_length
        self.mean, self.std = self.__normalize_mean_std()
        
        if isValSet_bool:
            dataframe = dataframe[len_test:]
            self.y = torch.tensor(dataframe[self.target].values).float()
            self.X = torch.tensor(dataframe[self.features].values).float()

        else:
            dataframe = dataframe[:len_test]
            self.y = torch.tensor(dataframe[self.target].values).float()
            self.X = torch.tensor(dataframe[self.features].values).float()


    def __normalize_mean_std(self):
        target_mean = self.dataframe[self.target].mean()
        target_stdev = self.dataframe[self.target].std()

        for i in self.dataframe.columns:
            mean = self.dataframe[i].mean()
            stdev = self.dataframe[i].std()

            self.dataframe[i] = np.divide((self.dataframe[i] - mean), stdev)
            self.dataframe[i] = np.divide((self.dataframe[i] - mean), stdev)

        return target_mean, target_stdev

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i] 

class StockPredictorLSTM(nn.Module):
    def __init__(self, 
                input_dim=4,  
                hidden_dim=32, 
                num_layers=2, 
                output_dim=1, 
                dropout=0.2):

        super().__init__()
        self.input_dim = input_dim  # this is the number of features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        x = self.dropout(x)
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out