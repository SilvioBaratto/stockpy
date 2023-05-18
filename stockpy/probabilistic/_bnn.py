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

class BayesianNNClassifier(ClassifierProb):
    """
    A class used to represent a Bayesian Neural Network (BNN) for classification tasks.
    This class inherits from the `ClassifierProb` class.

    ...

    Parameters
    ----------
    hidden_size:
        a list of integers that represents the number of nodes in each hidden layer or 
        a single integer that represents the number of nodes in a single hidden layer
    dropout:
        a float that represents the dropout rate for the dropout layers (default is 0.2)

    Attributes
    ----------
    model_type : str
        a string that represents the type of the model (default is "ffnn")

    Methods
    -------
    __init__(self, **kwargs):
        Initializes the BayesianNNClassifier object with given or default parameters.

    _init_model(self):
        Initializes the Bayesian Neural Network layers of the model based on configuration.

    forward(self, x_data: torch.Tensor, y_data: torch.Tensor=None) -> torch.Tensor:
        Defines the forward pass of the Bayesian Neural Network.

    _initSVI(self) -> pyro.infer.svi.SVI:
        Initializes Stochastic Variational Inference (SVI) for Bayesian Inference with an AutoNormal guide.
    """

    model_type = "ffnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initializes the BayesianNNClassifier object with given or default parameters.

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.prob.hidden_size, int):
            self.hidden_sizes = [cfg.prob.hidden_size]
        else:
            self.hidden_sizes = cfg.prob.hidden_size

        # Create the Bayesian Neural Network layers with Pyro's Module wrapper
        layers = []
        input_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layer = PyroModule[nn.Linear](input_size, hidden_size)
            layers.append(layer)
            layers.append(PyroModule[nn.ReLU]())
            layers.append(PyroModule[nn.Dropout](cfg.comm.dropout))
            input_size = hidden_size

        output_layer = PyroModule[nn.Linear](input_size, self.output_size)
        layers.append(output_layer)

        self.layers = PyroModule[nn.Sequential](*layers)

    def forward(self, x_data: torch.Tensor, y_data: torch.Tensor=None) -> torch.Tensor:
        # Ensures that the model has been initialized
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


class BayesianNNRegressor(RegressorProb):
    """
    A class used to represent a Bayesian Neural Network (BNN) for regression tasks.
    This class inherits from the `ClassifierProb` class.

    ...

    Parameters
    ----------
    hidden_size:
        a list of integers that represents the number of nodes in each hidden layer or 
        a single integer that represents the number of nodes in a single hidden layer
    dropout:
        a float that represents the dropout rate for the dropout layers (default is 0.2)

    Attributes
    ----------
    model_type : str
        a string that represents the type of the model (default is "ffnn")

    Methods
    -------
    __init__(self, **kwargs):
        Initializes the BayesianNNRegressor object with given or default parameters.

    _init_model(self):
        Initializes the Bayesian Neural Network layers of the model based on configuration.

    forward(self, x_data: torch.Tensor, y_data: torch.Tensor=None) -> torch.Tensor:
        Defines the forward pass of the Bayesian Neural Network.

    _initSVI(self) -> pyro.infer.svi.SVI:
        Initializes Stochastic Variational Inference (SVI) for Bayesian Inference with an AutoNormal guide.
    """

    model_type = "ffnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initializes the BayesianNNClassifier object with given or default parameters.

    def _init_model(self):
        # Check if hidden_sizes is a single integer and, if so, convert it to a list
        if isinstance(cfg.prob.hidden_size, int):
            self.hidden_sizes = [cfg.prob.hidden_size]
        else:
            self.hidden_sizes = cfg.prob.hidden_size

        # Create the Bayesian Neural Network layers with Pyro's Module wrapper
        layers = []
        input_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layer = PyroModule[nn.Linear](input_size, hidden_size)
            layers.append(layer)
            layers.append(PyroModule[nn.ReLU]())
            layers.append(PyroModule[nn.Dropout](cfg.comm.dropout))
            input_size = hidden_size

        output_layer = PyroModule[nn.Linear](input_size, self.output_size)
        layers.append(output_layer)

        self.layers = PyroModule[nn.Sequential](*layers)

    def forward(self, x_data: torch.Tensor, 
                y_data: torch.Tensor=None) -> torch.Tensor:
        """
        Computes the forward pass of the Bayesian Neural Network model using Pyro.

        :param x_data: the input data tensor
        :type x_data: torch.Tensor
        :param y_data: the target data tensor
        :type y_data: torch.Tensor

        :returns: the output tensor of the model
        :rtype: torch.Tensor
        """
        x =  self.layers(x_data)
        # use StudentT distribution instead of Normal
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
                test_dl : torch.utils.data.DataLoader
                ) -> torch.Tensor:

        return self._predictNN(test_dl)
    

    