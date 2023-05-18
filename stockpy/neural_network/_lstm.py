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

class LSTMClassifier(ClassifierNN):
    """
    A class used to represent a Long Short-Term Memory (LSTM) network for classification tasks. 
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
        Initializes the LSTMClassifier object with given or default parameters.

    _init_model(self):
        Initializes the LSTM and fully connected layers of the model based on configuration.

    forward(x: torch.Tensor) -> torch.Tensor:
        Defines the forward pass of the LSTM network.
    """

    model_type = "rnn"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initializes the LSTM neural network model with given or default parameters

    def _init_model(self):
        # Checks if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        # Initializes an empty ModuleList to hold the LSTM layers
        self.lstms = nn.ModuleList()
        input_size = self.input_size

        # Creates the LSTM layers of the neural network
        for hidden_size in self.hidden_sizes:
            self.lstms.append(nn.LSTM(input_size=input_size,  
                                      hidden_size=hidden_size, 
                                      num_layers=cfg.nn.num_layers, 
                                      batch_first=True))
            input_size = hidden_size

        # Initializes the fully connected (FC) layer which maps the last hidden state to the output
        self.fc = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LSTM network.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
        """

        # Ensures the model has been fitted before making predictions
        if not self.lstms:
            raise RuntimeError("You must call fit before calling predict")
        
        # Obtains the batch size
        batch_size = x.size(0)
        output = x

        # Iterates over LSTM layers, with each LSTM layer's output serving as input to the next
        for lstm in self.lstms:
            h0 = Variable(torch.zeros(cfg.nn.num_layers, 
                                      batch_size, 
                                      lstm.hidden_size)).to(cfg.training.device)
            c0 = Variable(torch.zeros(cfg.nn.num_layers, 
                                      batch_size, 
                                      lstm.hidden_size)).to(cfg.training.device)
            output, (hn, _) = lstm(output, (h0, c0))

        # Passes the last hidden state through the FC layer
        out = self.fc(output[:, -1, :])
        # Reshapes the output tensor
        out = out.view(-1,self.output_size)

        # Returns the output of the forward pass of the LSTM network
        return out
    
class LSTMRegressor(RegressorNN):
    """
    A class used to represent a Long Short-Term Memory (LSTM) network for regression tasks. 
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
        Initializes the LSTMRegressor object with given or default parameters.

    _init_model(self):
        Initializes the LSTM and fully connected layers of the model based on configuration.

    forward(x: torch.Tensor) -> torch.Tensor:
        Defines the forward pass of the LSTM network.
    """

    model_type = "rnn"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initializes the LSTM neural network model with given or default parameters

    def _init_model(self):
        # Checks if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        # Initializes an empty ModuleList to hold the LSTM layers
        self.lstms = nn.ModuleList()
        input_size = self.input_size

        # Creates the LSTM layers of the neural network
        for hidden_size in self.hidden_sizes:
            self.lstms.append(nn.LSTM(input_size=input_size,  
                                      hidden_size=hidden_size, 
                                      num_layers=cfg.nn.num_layers, 
                                      batch_first=True))
            input_size = hidden_size

        # Initializes the fully connected (FC) layer which maps the last hidden state to the output
        self.fc = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LSTM network.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
        """

        # Ensures the model has been fitted before making predictions
        if not self.lstms:
            raise RuntimeError("You must call fit before calling predict")
        
        # Obtains the batch size
        batch_size = x.size(0)
        output = x

        # Iterates over LSTM layers, with each LSTM layer's output serving as input to the next
        for lstm in self.lstms:
            h0 = Variable(torch.zeros(cfg.nn.num_layers, 
                                      batch_size, 
                                      lstm.hidden_size)).to(cfg.training.device)
            c0 = Variable(torch.zeros(cfg.nn.num_layers, 
                                      batch_size, 
                                      lstm.hidden_size)).to(cfg.training.device)
            output, (hn, _) = lstm(output, (h0, c0))

        # Passes the last hidden state through the FC layer
        out = self.fc(output[:, -1, :])
        # Reshapes the output tensor
        out = out.view(-1,self.output_size)

        # Returns the output of the forward pass of the LSTM network
        return out