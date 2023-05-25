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
    """
    A class used to represent a Convolutional Neural Network (CNN) for classification tasks.
    This class inherits from the `ClassifierNN` class.

    Attributes:
        model_type (str): A string that represents the type of the model (default is "cnn").

    Args:
        hidden_size (Union[int, List[int]]): A list of integers that represents the number of nodes in each hidden layer or
                                              a single integer that represents the number of nodes in a single hidden layer.
        num_filters (int): The number of filters in the convolutional layer.
        kernel_size (int): The size of the kernel in the convolutional layer.
        pool_size (int): The size of the pooling layer.

    Methods:
        __init__(self, **kwargs): Initializes the CNNClassifier object with given or default parameters.
        _init_model(self): Initializes the convolutional and fully connected layers of the model based on configuration.
        forward(x: torch.Tensor) -> torch.Tensor: Defines the forward pass of the CNN.
    """

    model_type = "cnn"

    def __init__(self, **kwargs):
        """
        Initializes the CNNClassifier object with given or default parameters.
        """
        super().__init__(**kwargs)

    def _init_model(self):
        """
        Initializes the convolutional and fully connected layers of the model based on configuration.
        """
        # Create the convolutional layers

        # Check if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        # Initializes a list to store the layers of the neural network
        layers = [nn.Conv1d(1, cfg.nn.num_filters, cfg.nn.kernel_size),  # 1D convolutional layer
                  nn.ReLU(),  # Activation function
                  nn.MaxPool1d(cfg.nn.pool_size),  # Max pooling layer
                  nn.Flatten()]  # Flatten layer for transforming the output for use in FC layers

        # Calculates the input size for the first FC layer after flattening
        current_input_size = cfg.nn.num_filters * ((self.input_size - cfg.nn.kernel_size + 1) \
                                                    // cfg.nn.pool_size)

        # Creates the FC layers of the neural network
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))  # Linear (FC) layer
            layers.append(nn.ReLU())  # Activation function
            layers.append(nn.Dropout(cfg.comm.dropout))  # Dropout layer for regularization
            current_input_size = hidden_size

        # Adds the output FC layer
        layers.append(nn.Linear(current_input_size, self.output_size))

        # Creates the neural network as a sequential model based on the layers list
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
        """
        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        return self.layers(x)


class CNNRegressor(RegressorNN):
    """
    A class used to represent a Convolutional Neural Network (CNN) for regression tasks.
    This class inherits from the `RegressorNN` class.

    Attributes:
        model_type (str): A string that represents the type of the model (default is "cnn").

    Args:
        hidden_size (Union[int, List[int]]): A list of integers that represents the number of nodes in each hidden layer or
                                              a single integer that represents the number of nodes in a single hidden layer.
        num_filters (int): The number of filters in the convolutional layer.
        kernel_size (int): The size of the kernel in the convolutional layer.
        pool_size (int): The size of the pooling layer.

    Methods:
        __init__(self, **kwargs): Initializes the CNNRegressor object with given or default parameters.
        _init_model(self): Initializes the convolutional and fully connected layers of the model based on configuration.
        forward(x: torch.Tensor) -> torch.Tensor: Defines the forward pass of the CNN.
    """

    model_type = "cnn"

    def __init__(self, **kwargs):
        """
        Initializes the CNNRegressor object with given or default parameters.
        """
        super().__init__(**kwargs)

    def _init_model(self):
        """
        Initializes the convolutional and fully connected layers of the model based on configuration.
        """
        # Create the convolutional layers

        # Check if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        # Initializes a list to store the layers of the neural network
        layers = [nn.Conv1d(1, cfg.nn.num_filters, cfg.nn.kernel_size),  # 1D convolutional layer
                  nn.ReLU(),  # Activation function
                  nn.MaxPool1d(cfg.nn.pool_size),  # Max pooling layer
                  nn.Flatten()]  # Flatten layer for transforming the output for use in FC layers

        # Calculates the input size for the first FC layer after flattening
        current_input_size = cfg.nn.num_filters * ((self.input_size - cfg.nn.kernel_size + 1) \
                                                    // cfg.nn.pool_size)

        # Creates the FC layers of the neural network
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))  # Linear (FC) layer
            layers.append(nn.ReLU())  # Activation function
            layers.append(nn.Dropout(cfg.comm.dropout))  # Dropout layer for regularization
            current_input_size = hidden_size

        # Adds the output FC layer
        layers.append(nn.Linear(current_input_size, self.output_size))

        # Creates the neural network as a sequential model based on the layers list
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
        """
        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        return self.layers(x)