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
    """
    A class used to represent a Multilayer Perceptron (MLP) for classification tasks.
    This class inherits from the `ClassifierNN` class.

    Attributes:
        model_type (str): A string that represents the type of the model (default is "ffnn").

    Args:
        hidden_size (Union[int, List[int]]): A list of integers that represents the number of nodes in each hidden layer or
                                              a single integer that represents the number of nodes in a single hidden layer.
        dropout (float): The dropout probability (default is 0.2).

    Methods:
        __init__(self, **kwargs): Initializes the MLPClassifier object with given or default parameters.
        _init_model(self): Initializes the layers of the neural network based on configuration.
        forward(x: torch.Tensor) -> torch.Tensor: Defines the forward pass of the neural network.
    """

    model_type = "ffnn"

    def __init__(self, **kwargs):
        """
        Initializes the MLPClassifier object with given or default parameters.
        """
        super().__init__(**kwargs)

    def _init_model(self):
        """
        Initializes the layers of the neural network based on configuration.
        """
        # Checks if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        layers = []
        input_size = self.input_size
        # Creates the layers of the neural network
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.comm.dropout))
            input_size = hidden_size

        # Appends the output layer to the neural network
        layers.append(nn.Linear(input_size, self.output_size))
        # Stacks all the layers into a sequence
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
        """
        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        # Returns the output of the forward pass of the neural network
        return self.layers(x)
class MLPRegressor(RegressorNN):
    """
    A class used to represent a Multilayer Perceptron (MLP) for regression tasks.
    This class inherits from the `RegressorNN` class.

    Attributes:
        model_type (str): A string that represents the type of the model (default is "ffnn").

    Args:
        hidden_size (Union[int, List[int]]): A list of integers that represents the number of nodes in each hidden layer or
                                              a single integer that represents the number of nodes in a single hidden layer.
        dropout (float): The dropout probability (default is 0.2).

    Methods:
        __init__(self, **kwargs): Initializes the MLPRegressor object with given or default parameters.
        _init_model(self): Initializes the layers of the neural network based on configuration.
        forward(x: torch.Tensor) -> torch.Tensor: Defines the forward pass of the neural network.
    """

    model_type = "ffnn"

    def __init__(self, **kwargs):
        """
        Initializes the MLPRegressor object with given or default parameters.
        """
        super().__init__(**kwargs)

    def _init_model(self):
        """
        Initializes the layers of the neural network based on configuration.
        """
        # Checks if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        layers = []
        input_size = self.input_size
        # Creates the layers of the neural network
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.comm.dropout))
            input_size = hidden_size

        # Appends the output layer to the neural network
        layers.append(nn.Linear(input_size, self.output_size))
        # Stacks all the layers into a sequence
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
        """
        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        # Returns the output of the forward pass of the neural network
        return self.layers(x)      

