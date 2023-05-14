from abc import ABCMeta, abstractmethod
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Union, Tuple
import pandas as pd
import numpy as np
from ._base import ClassifierNN
from ._base import RegressorNN
from ..config import Config as cfg

class BiLSTMClassifier(ClassifierNN):

    model_type = "rnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        self.lstms = nn.ModuleList()
        input_size = self.input_size

        for hidden_size in self.hidden_sizes:
            self.lstms.append(nn.LSTM(input_size=input_size,  
                                      hidden_size=hidden_size, 
                                      num_layers=1, 
                                      batch_first=True))
            input_size = hidden_size

        self.fc = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.lstms:
            raise RuntimeError("You must call fit before calling predict")
        
        batch_size = x.size(0)
        output = x

        for lstm in self.lstms:
            h0 = Variable(torch.zeros(1, batch_size, lstm.hidden_size)).to(cfg.training.device)
            c0 = Variable(torch.zeros(1, batch_size, lstm.hidden_size)).to(cfg.training.device)
            output, (hn, _) = lstm(output, (h0, c0))

        out = self.fc(output[:, -1, :])
        out = out.view(-1,self.output_size)

        return out
    
class BiLSTMRegressor(RegressorNN):

    model_type = "rnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        self.bilstms = nn.ModuleList()
        input_size = self.input_size

        for hidden_size in self.hidden_sizes:
            self.bilstms.append(nn.LSTM(input_size=input_size,  
                                        hidden_size=hidden_size, 
                                        num_layers=1, 
                                        bidirectional=True, 
                                        batch_first=True))
            input_size = hidden_size * 2  # times 2 because of bidirectional

        self.fc = nn.Linear(self.hidden_sizes[-1] * 2, self.output_size)  # times 2 because of bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.bilstms:
            raise RuntimeError("You must call fit before calling predict")
        
        batch_size = x.size(0)
        output = x

        for bilstm in self.bilstms:
            h0 = Variable(torch.zeros(2, batch_size, bilstm.hidden_size)).to(cfg.training.device)  # times 2 because of bidirectional
            c0 = Variable(torch.zeros(2, batch_size, bilstm.hidden_size)).to(cfg.training.device)  # times 2 because of bidirectional
            output, (hn, _) = bilstm(output, (h0, c0))

        out = self.fc(output[:, -1, :])
        out = out.view(-1,self.output_size)

        return out
