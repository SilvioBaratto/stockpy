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
    """
    A class used to represent a Gated Recurrent Unit (GRU) network for classification tasks. 
    This class inherits from the `ClassifierNN` class.

    ...

    Parameters
    ----------
    hidden_size:
        a list of integers that represents the number of nodes in each hidden layer or 
        a single integer that represents the number of nodes in a single hidden layer
    num_layers:
        the number of recurrent layers (default is 1)

    Attributes
    ----------
    model_type : str
        a string that represents the type of the model (default is "rnn")

    Methods
    -------
    __init__(self, **kwargs):
        Initializes the GRUClassifier object with given or default parameters.

    _init_model(self):
        Initializes the GRU and fully connected layers of the model based on configuration.

    forward(x: torch.Tensor) -> torch.Tensor:
        Defines the forward pass of the GRU network.
    """

    model_type = "rnn"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initializes the GRU neural network model with given or default parameters

    def _init_model(self):
        # Checks if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        # Initializes an empty ModuleList to hold the GRU layers
        self.grus = nn.ModuleList()
        input_size = self.input_size

        # Creates the GRU layers of the neural network
        for hidden_size in self.hidden_sizes:
            self.grus.append(nn.GRU(input_size=input_size,  
                                    hidden_size=hidden_size, 
                                    num_layers=cfg.nn.num_layers, 
                                    batch_first=True))
            input_size = hidden_size

        # Initializes the fully connected (FC) layer which maps the last hidden state to the output
        self.fc = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the GRU network.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
        """
        # Ensures the model has been fitted before making predictions
        if not self.grus:
            raise RuntimeError("You must call fit before calling predict")
        
        # Obtains the batch size
        batch_size = x.size(0)
        output = x

        # Iterates over GRU layers, with each GRU layer's output serving as input to the next
        for gru in self.grus:
            h0 = Variable(torch.zeros(cfg.nn.num_layers, 
                                      batch_size, 
                                      gru.hidden_size)).to(cfg.training.device)
            output, hn = gru(output, h0)

        # Passes the last hidden state through the FC layer
        out = self.fc(output[:, -1, :])
        # Reshapes the output tensor
        out = out.view(-1,self.output_size)

        # Returns the output of the forward pass of the GRU network
        return out
    
class GRURegressor(RegressorNN):
    """
    A class used to represent a Gated Recurrent Unit (GRU) network for regression tasks. 
    This class inherits from the `RegressorNN` class.

    ...

    Parameters
    ----------
    hidden_size:
        a list of integers that represents the number of nodes in each hidden layer or 
        a single integer that represents the number of nodes in a single hidden layer
    num_layers:
        the number of recurrent layers (default is 1)

    Attributes
    ----------
    model_type : str
        a string that represents the type of the model (default is "rnn")

    Methods
    -------
    __init__(self, **kwargs):
        Initializes the GRURegressor object with given or default parameters.

    _init_model(self):
        Initializes the GRU and fully connected layers of the model based on configuration.

    forward(x: torch.Tensor) -> torch.Tensor:
        Defines the forward pass of the GRU network.
    """

    model_type = "rnn"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initializes the GRU neural network model with given or default parameters

    def _init_model(self):
        # Checks if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        # Initializes an empty ModuleList to hold the GRU layers
        self.grus = nn.ModuleList()
        input_size = self.input_size

        # Creates the GRU layers of the neural network
        for hidden_size in self.hidden_sizes:
            self.grus.append(nn.GRU(input_size=input_size,  
                                    hidden_size=hidden_size, 
                                    num_layers=cfg.nn.num_layers, 
                                    batch_first=True))
            input_size = hidden_size

        # Initializes the fully connected (FC) layer which maps the last hidden state to the output
        self.fc = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the GRU network.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
        """

        # Ensures the model has been fitted before making predictions
        if not self.grus:
            raise RuntimeError("You must call fit before calling predict")
        
        # Obtains the batch size
        batch_size = x.size(0)
        output = x

        # Iterates over GRU layers, with each GRU layer's output serving as input to the next
        for gru in self.grus:
            h0 = Variable(torch.zeros(cfg.nn.num_layers, 
                                      batch_size, 
                                      gru.hidden_size)).to(cfg.training.device)
            output, hn = gru(output, h0)

        # Passes the last hidden state through the FC layer
        out = self.fc(output[:, -1, :])
        # Reshapes the output tensor
        out = out.view(-1,self.output_size)

        # Returns the output of the forward pass of the GRU network
        return out
        
        
