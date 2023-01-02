import argparse
import datetime
import hashlib
import os
import shutil
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class LSTM(nn.Module):
    def __init__(self, 
                input_size=4,  
                hidden_size=32, 
                num_layers=2, 
                output_dim=1, 
                dropout=0.2,
                seq_length=30
                ):

        super(LSTM, self).__init__()
        self.input_size = input_size # this is the number of features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = seq_length

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.dropout = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, output_dim)
        # self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        # hn = hn.view(-1, self.hidden_size)
        # x = self.dropout(x)
        # out = self.relu(hn[0]).flatten()
        # out = self.fc(hn[0]).flatten() #first Dense
        
        out = self.relu(hn[0])
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
       
        out = out.view(-1,1)

        return out

"""

class LSTM(nn.Module):
    def __init__(self, 
                input_size=4,  
                hidden_size=32, 
                num_layers=2, 
                output_dim=1, 
                # dropout=0.2,
                seq_length=30
                ):

        super().__init__()
        self.input_size = input_size  # this is the number of features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True
                            )

        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, output_dim)
        # self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output

        return out
        
"""
        
