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

class BiGRUClassifier(ClassifierNN):

    model_type = "rnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        self.bigrus = nn.ModuleList()
        input_size = self.input_size

        for hidden_size in self.hidden_sizes:
            self.bigrus.append(nn.GRU(input_size=input_size,  
                                      hidden_size=hidden_size, 
                                      num_layers=1, 
                                      bidirectional=True,
                                      batch_first=True))
            input_size = hidden_size * 2  # times 2 because of bidirectional

        self.fc = nn.Linear(self.hidden_sizes[-1] * 2, self.output_size)  # times 2 because of bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.bigrus:
            raise RuntimeError("You must call fit before calling predict")
        
        batch_size = x.size(0)
        output = x

        for bigru in self.bigrus:
            h0 = Variable(torch.zeros(2, batch_size, bigru.hidden_size)).to(cfg.training.device)  # times 2 because of bidirectional
            output, hn = bigru(output, h0)

        out = self.fc(output[:, -1, :])
        out = out.view(-1,self.output_size)

        return out
    
class BiGRURegressor(RegressorNN):

    model_type = "rnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        self.bigrus = nn.ModuleList()
        input_size = self.input_size

        for hidden_size in self.hidden_sizes:
            self.bigrus.append(nn.GRU(input_size=input_size,  
                                      hidden_size=hidden_size, 
                                      num_layers=1, 
                                      bidirectional=True,
                                      batch_first=True))
            input_size = hidden_size * 2  # times 2 because of bidirectional

        self.fc = nn.Linear(self.hidden_sizes[-1] * 2, self.output_size)  # times 2 because of bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.bigrus:
            raise RuntimeError("You must call fit before calling predict")
        
        batch_size = x.size(0)
        output = x

        for bigru in self.bigrus:
            h0 = Variable(torch.zeros(2, batch_size, bigru.hidden_size)).to(cfg.training.device)  # times 2 because of bidirectional
            output, hn = bigru(output, h0)

        out = self.fc(output[:, -1, :])
        out = out.view(-1,self.output_size)

        return out