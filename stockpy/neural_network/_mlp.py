from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from typing import Union, Tuple
import pandas as pd
import numpy as np
from ._base import ClassifierNN
from ._base import RegressorNN
from ..config import Config as cfg

class MLPClassifier(ClassifierNN):

    model_type = "ffnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Initializes the MLP neural network model.

        :param args: The arguments to configure the model.
        :type args: ModelArgs
        """

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        layers = []
        input_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.comm.dropout))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        :param x: The input tensor.
        :type x: torch.Tensor

        :returns: The output tensor, corresponding to the predicted target variable(s).
        :rtype: torch.Tensor
        """
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        return self.layers(x)  
    
class MLPRegressor(RegressorNN):

    model_type = "ffnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Initializes the MLP neural network model.

        :param args: The arguments to configure the model.
        :type args: ModelArgs
        """

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        layers = []
        input_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.comm.dropout))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        :param x: The input tensor.
        :type x: torch.Tensor

        :returns: The output tensor, corresponding to the predicted target variable(s).
        :rtype: torch.Tensor
        """
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        return self.layers(x)        

