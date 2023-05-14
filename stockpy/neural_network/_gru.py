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

class GRUClassifier(ClassifierNN):

    model_type = "rnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        self.grus = nn.ModuleList()
        input_size = self.input_size

        for hidden_size in self.hidden_sizes:
            self.grus.append(nn.GRU(input_size=input_size,  
                                    hidden_size=hidden_size, 
                                    num_layers=1, 
                                    batch_first=True))
            input_size = hidden_size

        self.fc = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.grus:
            raise RuntimeError("You must call fit before calling predict")
        
        batch_size = x.size(0)
        output = x

        for gru in self.grus:
            h0 = Variable(torch.zeros(1, batch_size, gru.hidden_size)).to(cfg.training.device)
            output, hn = gru(output, h0)

        out = self.fc(output[:, -1, :])
        out = out.view(-1,self.output_size)

        return out
    
class GRURegressor(RegressorNN):

    model_type = "rnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        self.grus = nn.ModuleList()
        input_size = self.input_size

        for hidden_size in self.hidden_sizes:
            self.grus.append(nn.GRU(input_size=input_size,  
                                    hidden_size=hidden_size, 
                                    num_layers=1, 
                                    batch_first=True))
            input_size = hidden_size

        self.fc = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.grus:
            raise RuntimeError("You must call fit before calling predict")
        
        batch_size = x.size(0)
        output = x

        for gru in self.grus:
            h0 = Variable(torch.zeros(1, batch_size, gru.hidden_size)).to(cfg.training.device)
            output, hn = gru(output, h0)

        out = self.fc(output[:, -1, :])
        out = out.view(-1,self.output_size)

        return out
        
        
