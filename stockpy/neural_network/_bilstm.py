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
    """
    A class used to represent a Bidirectional Long Short-Term Memory (BiLSTM) network for classification tasks. 
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
        Initializes the BiLSTMClassifier object with given or default parameters.

    _init_model(self):
        Initializes the BiLSTM layers and fully connected layer of the model based on configuration.

    forward(x: torch.Tensor) -> torch.Tensor:
        Defines the forward pass of the BiLSTM network.
    """

    model_type = "rnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initializes the BiLSTM object with given or default parameters.

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        # Initializes an empty module list for the BiLSTM layers
        self.bilstms = nn.ModuleList()
        input_size = self.input_size

        # Iterates through the hidden sizes and creates BiLSTM layers accordingly
        for hidden_size in self.hidden_sizes:
            self.bilstms.append(nn.LSTM(input_size=input_size,  
                                        hidden_size=hidden_size, 
                                        num_layers=cfg.nn.num_layers, 
                                        bidirectional=True, 
                                        batch_first=True))
            
            # Multiplies by 2 for the next input size because the LSTM is bidirectional
            input_size = hidden_size * 2 

        # The final fully connected layer's input size is also doubled because the LSTM is bidirectional
        self.fc = nn.Linear(self.hidden_sizes[-1] * 2, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensures that the model has been initialized
        if not self.bilstms:
            raise RuntimeError("You must call fit before calling predict")
        
        batch_size = x.size(0)
        output = x

        # Applies each BiLSTM layer on the input tensor
        for bilstm in self.bilstms:
            h0 = Variable(torch.zeros(cfg.nn.num_layers * 2, 
                                      batch_size, 
                                      bilstm.hidden_size)).to(cfg.training.device)  # times 2 because of bidirectional
            c0 = Variable(torch.zeros(cfg.nn.num_layers * 2, 
                                      batch_size, 
                                      bilstm.hidden_size)).to(cfg.training.device)  # times 2 because of bidirectional
            output, (hn, _) = bilstm(output, (h0, c0))

        # Applies the final fully connected layer
        out = self.fc(output[:, -1, :])
        out = out.view(-1, self.output_size)

        return out
    
class BiLSTMRegressor(RegressorNN):
    """
    A class used to represent a Bidirectional Long Short-Term Memory (BiLSTM) network for regression tasks.
    This class inherits from the `RegressorNN` class.

    Attributes:
        model_type (str): A string that represents the type of the model (default is "rnn").

    Args:
        hidden_size (Union[int, List[int]]): A list of integers that represents the number of nodes in each hidden layer or
                                              a single integer that represents the number of nodes in a single hidden layer.
        num_layers (int): The number of recurrent layers (default is 1).

    Methods:
        __init__(self, **kwargs): Initializes the BiLSTMRegressor object with given or default parameters.
        _init_model(self): Initializes the BiLSTM layers and fully connected layer of the model based on configuration.
        forward(x: torch.Tensor) -> torch.Tensor: Defines the forward pass of the BiLSTM network.
    """

    model_type = "rnn"

    def __init__(self, **kwargs):
        """
        Initializes the BiLSTMRegressor object with given or default parameters.
        """
        super().__init__(**kwargs)

    def _init_model(self):
        """
        Initializes the BiLSTM layers and fully connected layer of the model based on configuration.
        """
        # Check if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        # Initializes an empty module list for the BiLSTM layers
        self.bilstms = nn.ModuleList()
        input_size = self.input_size

        # Iterates through the hidden sizes and creates BiLSTM layers accordingly
        for hidden_size in self.hidden_sizes:
            self.bilstms.append(nn.LSTM(input_size=input_size,
                                        hidden_size=hidden_size,
                                        num_layers=cfg.nn.num_layers,
                                        bidirectional=True,
                                        batch_first=True))

            # Multiplies by 2 for the next input size because the LSTM is bidirectional
            input_size = hidden_size * 2

        # The final fully connected layer's input size is also doubled because the LSTM is bidirectional
        self.fc = nn.Linear(self.hidden_sizes[-1] * 2, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the BiLSTM network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        Raises:
            RuntimeError: If the model has not been initialized by calling the fit method before calling predict.
        """
        # Ensures that the model has been initialized
        if not self.bilstms:
            raise RuntimeError("You must call fit before calling predict")
        
        batch_size = x.size(0)
        output = x

        # Applies each BiLSTM layer on the input tensor
        for bilstm in self.bilstms:
            h0 = Variable(torch.zeros(cfg.nn.num_layers * 2, 
                                      batch_size, 
                                      bilstm.hidden_size)).to(cfg.training.device)  # times 2 because of bidirectional
            c0 = Variable(torch.zeros(cfg.nn.num_layers * 2, 
                                      batch_size, 
                                      bilstm.hidden_size)).to(cfg.training.device)  # times 2 because of bidirectional
            output, (hn, _) = bilstm(output, (h0, c0))

        # Applies the final fully connected layer
        out = self.fc(output[:, -1, :])
        out = out.view(-1, self.output_size)

        return out
