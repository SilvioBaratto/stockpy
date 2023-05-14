from abc import ABCMeta, abstractmethod
import os
import torch
import torch.nn as nn
from typing import Union, Tuple
import pandas as pd
import numpy as np
from ._base import ClassifierNN
from ._base import RegressorNN
from ..config import Config as cfg

class CNNClassifier(ClassifierNN):

    model_type = "cnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Initializes the MLP neural network model.

        :param args: The arguments to configure the model.
        :type args: ModelArgs
        """

    def _init_model(self):
        # Create the convolutional layers
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size
            
        layers = [nn.Conv1d(1, cfg.nn.num_filters, cfg.nn.kernel_size),
                  nn.ReLU(),
                  nn.MaxPool1d(cfg.nn.pool_size),
                  nn.Flatten()]
        
        current_input_size = cfg.nn.num_filters * ((self.input_size - cfg.nn.kernel_size + 1) // cfg.nn.pool_size)

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.comm.dropout))
            current_input_size = hidden_size

        layers.append(nn.Linear(current_input_size, self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        return self.layers(x)
    
class CNNRegressor(RegressorNN):

    model_type = "cnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Initializes the MLP neural network model.

        :param args: The arguments to configure the model.
        :type args: ModelArgs
        """

    def _init_model(self):
        # Create the convolutional layers
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size
            
        layers = [nn.Conv1d(1, cfg.nn.num_filters, cfg.nn.kernel_size),
                  nn.ReLU(),
                  nn.MaxPool1d(cfg.nn.pool_size),
                  nn.Flatten()]
        
        current_input_size = cfg.nn.num_filters * ((self.input_size - cfg.nn.kernel_size + 1) // cfg.nn.pool_size)

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.comm.dropout))
            current_input_size = hidden_size

        layers.append(nn.Linear(cfg.nn.hidden_size[-1], self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        return self.layers(x)
