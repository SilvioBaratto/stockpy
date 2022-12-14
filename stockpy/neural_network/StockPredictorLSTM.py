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
