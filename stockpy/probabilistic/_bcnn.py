from abc import abstractmethod, ABCMeta
import os
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoNormal
from pyro.infer import (
    SVI,
    Trace_ELBO,
    Predictive,
    TraceMeanField_ELBO
)
from typing import Union, Tuple
import pandas as pd
import numpy as np
from ._base import ClassifierProb
from ._base import RegressorProb
from ..config import Config as cfg

class BCNNClassifier(ClassifierProb):
    """
    A class used to represent a Bayesian Convolutional Neural Network (CNN) 
    for probabilistic classification tasks. This class inherits from the 
    `ClassifierProb` class.

    ...

    Args:
        hidden_size: A list of integers that represents the number of nodes in each hidden layer or 
            a single integer that represents the number of nodes in a single hidden layer.
        num_filters: The number of filters in the convolutional layer.
        kernel_size: The size of the kernel in the convolutional layer.
        pool_size: The size of the pooling layer.
        dropout: The dropout rate for regularization.

    Attributes:
        model_type: A string that represents the type of the model (default is "cnn").

    Methods:
        __init__(self, **kwargs): Initializes the BayesianCNNClassifier object with given or default parameters.
        _init_model(self): Initializes the Convolutional layers and fully connected layers of the model 
            based on configuration. This model uses PyroModule wrappers for the layers 
            to enable Bayesian inference.
        forward(self, x_data: torch.Tensor, y_data: torch.Tensor=None) -> torch.Tensor: Defines the forward pass 
            of the Bayesian CNN, and optionally observes the output if ground truth `y_data` is provided.
        _initSVI(self) -> pyro.infer.svi.SVI: Initializes Stochastic Variational Inference (SVI) for the model.
            Defines the guide function to be a Normal distribution that learns to approximate the posterior, 
            and uses Mean Field ELBO as the variational loss.
    """

    model_type = "cnn"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initializes the BayesianCNNClassifier object with given or default parameters.

    def _init_model(self):
        # Create the convolutional layers

        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        # Create the Bayesian CNN layers with Pyro's Module wrapper
        layers = [PyroModule[nn.Conv1d](1, cfg.nn.num_filters, cfg.nn.kernel_size),  # 1D convolutional layer
                  PyroModule[nn.ReLU](),  # Activation function
                  PyroModule[nn.MaxPool1d](cfg.nn.pool_size),  # Max pooling layer
                  PyroModule[nn.Flatten]()]  # Flatten layer for transforming the output for use in FC layers

        # Calculate the input size for the first FC layer after flattening
        current_input_size = cfg.nn.num_filters * ((self.input_size - cfg.nn.kernel_size + 1) // cfg.nn.pool_size)

        # Create the FC layers of the neural network with Pyro's Module wrapper
        for hidden_size in self.hidden_sizes:
            layers.append(PyroModule[nn.Linear](current_input_size, hidden_size))  # Linear (FC) layer
            layers.append(PyroModule[nn.ReLU]())  # Activation function
            layers.append(PyroModule[nn.Dropout](cfg.comm.dropout))  # Dropout layer for regularization
            current_input_size = hidden_size

        # Add the output FC layer with Pyro's Module wrapper
        layers.append(PyroModule[nn.Linear](current_input_size, self.output_size))

        # Create the neural network as a sequential model based on the layers list with Pyro's Module wrapper
        self.layers = PyroModule[nn.Sequential](*layers)

    def forward(self, x_data: torch.Tensor, y_data: torch.Tensor=None) -> torch.Tensor:
        # Ensure that the model has been initialized
        if self.layers is None:
            raise Exception("Model has not been initialized.")

        x = self.layers(x_data)

        # Apply softmax to the output to get class probabilities
        x = nn.functional.softmax(x, dim=1)

        # Pyro uses this to sample the observed data for probabilistic models
        with pyro.plate("data", x_data.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(probs=x).to_event(1), obs=y_data)

        return x

    def _initSVI(self) -> pyro.infer.svi.SVI:
        # Pyro's Stochastic Variational Inference (SVI) is used for Bayesian inference
        self.guide = AutoNormal(self.forward)
        return SVI(model=self.forward,
                   guide=self.guide,
                   optim=self.optimizer,
                   loss=TraceMeanField_ELBO())


class BCNNRegressor(RegressorProb):
    """
    A class used to represent a Bayesian Convolutional Neural Network (CNN) 
    for probabilistic regression tasks. This class inherits from the 
    `ClassifierProb` class.

    ...

    Args:
        hidden_size: A list of integers that represents the number of nodes in each hidden layer or 
            a single integer that represents the number of nodes in a single hidden layer.
        num_filters: The number of filters in the convolutional layer.
        kernel_size: The size of the kernel in the convolutional layer.
        pool_size: The size of the pooling layer.
        dropout: The dropout rate for regularization.

    Attributes:
        model_type: A string that represents the type of the model (default is "cnn").

    Methods:
        __init__(self, **kwargs): Initializes the BayesianCNNRegressor object with given or default parameters.
        _init_model(self): Initializes the Convolutional layers and fully connected layers of the model 
            based on configuration. This model uses PyroModule wrappers for the layers 
            to enable Bayesian inference.
        forward(self, x_data: torch.Tensor, y_data: torch.Tensor=None) -> torch.Tensor: Defines the forward pass 
            of the Bayesian CNN, and optionally observes the output if ground truth `y_data` is provided.
        _initSVI(self) -> pyro.infer.svi.SVI: Initializes Stochastic Variational Inference (SVI) for the model.
            Defines the guide function to be a Normal distribution that learns to approximate the posterior, 
            and uses Mean Field ELBO as the variational loss.
    """

    model_type = "cnn"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initializes the BayesianCNNClassifier object with given or default parameters.

    def _init_model(self):
        # Create the convolutional layers

        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.nn.hidden_size, int):
            self.hidden_sizes = [cfg.nn.hidden_size]
        else:
            self.hidden_sizes = cfg.nn.hidden_size

        # Create the Bayesian CNN layers with Pyro's Module wrapper
        layers = [PyroModule[nn.Conv1d](1, cfg.nn.num_filters, cfg.nn.kernel_size),  # 1D convolutional layer
                  PyroModule[nn.ReLU](),  # Activation function
                  PyroModule[nn.MaxPool1d](cfg.nn.pool_size),  # Max pooling layer
                  PyroModule[nn.Flatten]()]  # Flatten layer for transforming the output for use in FC layers

        # Calculate the input size for the first FC layer after flattening
        current_input_size = cfg.nn.num_filters * ((self.input_size - cfg.nn.kernel_size + 1) // cfg.nn.pool_size)

        # Create the FC layers of the neural network with Pyro's Module wrapper
        for hidden_size in self.hidden_sizes:
            layers.append(PyroModule[nn.Linear](current_input_size, hidden_size))  # Linear (FC) layer
            layers.append(PyroModule[nn.ReLU]())  # Activation function
            layers.append(PyroModule[nn.Dropout](cfg.comm.dropout))  # Dropout layer for regularization
            current_input_size = hidden_size

        # Add the output FC layer with Pyro's Module wrapper
        layers.append(PyroModule[nn.Linear](current_input_size, self.output_size))

        # Create the neural network as a sequential model based on the layers list with Pyro's Module wrapper
        self.layers = PyroModule[nn.Sequential](*layers)

    def forward(self, x_data: torch.Tensor,
                y_data: torch.Tensor=None) -> torch.Tensor:

        # Ensure that the model has been initialized
        if self.layers is None:
            raise Exception("Model has not been initialized.")

        x = self.layers(x_data)

        # Use StudentT distribution instead of Normal
        df = pyro.sample("df", dist.Exponential(1.))
        scale = pyro.sample("scale", dist.HalfCauchy(2.5))
        with pyro.plate("data", x_data.shape[0]):
            obs = pyro.sample("obs", dist.StudentT(df, x, scale).to_event(1),
                              obs=y_data)

        return x

    def _initSVI(self) -> pyro.infer.svi.SVI:
        # Pyro's Stochastic Variational Inference (SVI) is used for Bayesian inference
        self.guide = AutoNormal(self.forward)
        return SVI(model=self.forward,
                   guide=self.guide,
                   optim=self.optimizer,
                   loss=TraceMeanField_ELBO())

    def _predict(self,
                 test_dl: torch.utils.data.DataLoader
                 ) -> torch.Tensor:

        return self._predictNN(test_dl)