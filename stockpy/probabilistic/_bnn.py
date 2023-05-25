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

class BNNClassifier(ClassifierProb):
    """
    A class used to represent a Bayesian Neural Network (BNN) for classification tasks.
    This class inherits from the `ClassifierProb` class.

    Attributes:
        hidden_size (list[int] or int): A list of integers representing the number of nodes in each hidden layer
            or a single integer representing the number of nodes in a single hidden layer.
        dropout (float): The dropout rate for the dropout layers (default is 0.2).
        model_type (str): A string representing the type of the model (default is "ffnn").

    Methods:
        __init__(self, **kwargs): Initializes the BayesianNNClassifier object with given or default parameters.
        _init_model(self): Initializes the Bayesian Neural Network layers of the model based on configuration.
        forward(self, x_data: torch.Tensor, y_data: torch.Tensor=None) -> torch.Tensor:
            Defines the forward pass of the Bayesian Neural Network.
        _initSVI(self) -> pyro.infer.svi.SVI: Initializes Stochastic Variational Inference (SVI)
            for Bayesian Inference with an AutoNormal guide.
    """

    model_type = "ffnn"
   
    def __init__(self, **kwargs):
        """
        Initializes the BayesianNNClassifier object with given or default parameters.
        """
        super().__init__(**kwargs)

    def _init_model(self):
        """
        Initializes the Bayesian Neural Network layers of the model based on configuration.
        """
        if isinstance(cfg.prob.hidden_size, int):
            self.hidden_sizes = [cfg.prob.hidden_size]
        else:
            self.hidden_sizes = cfg.prob.hidden_size

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
        """
        Defines the forward pass of the Bayesian Neural Network.

        Args:
            x_data (torch.Tensor): The input data tensor.
            y_data (torch.Tensor): The target data tensor.

        Returns:
            torch.Tensor: The output tensor of the model.
        """
        if self.layers is None:
            raise Exception("Model has not been initialized.")
        
        x = self.layers(x_data)
        
        x = nn.functional.softmax(x, dim=1)

        with pyro.plate("data", x_data.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(probs=x).to_event(1), obs=y_data)
            
        return x
    
    def _initSVI(self) -> pyro.infer.svi.SVI:
        """
        Initializes Stochastic Variational Inference (SVI) for Bayesian Inference with an AutoNormal guide.

        Returns:
            pyro.infer.svi.SVI: The SVI object.
        """
        self.guide = AutoNormal(self.forward)
        return SVI(model=self.forward,
                   guide=self.guide,
                   optim=self.optimizer, 
                   loss=TraceMeanField_ELBO())


class BNNRegressor(RegressorProb):
    """
    A class used to represent a Bayesian Neural Network (BNN) for regression tasks.
    This class inherits from the `ClassifierProb` class.

    Attributes:
        hidden_size (list[int] or int): A list of integers representing the number of nodes in each hidden layer
            or a single integer representing the number of nodes in a single hidden layer.
        dropout (float): The dropout rate for the dropout layers (default is 0.2).
        model_type (str): A string representing the type of the model (default is "ffnn").

    Methods:
        __init__(self, **kwargs): Initializes the BayesianNNRegressor object with given or default parameters.
        _init_model(self): Initializes the Bayesian Neural Network layers of the model based on configuration.
        forward(self, x_data: torch.Tensor, y_data: torch.Tensor=None) -> torch.Tensor:
            Defines the forward pass of the Bayesian Neural Network.
        _initSVI(self) -> pyro.infer.svi.SVI: Initializes Stochastic Variational Inference (SVI)
            for Bayesian Inference with an AutoNormal guide.
    """

    model_type = "ffnn"
   
    def __init__(self, **kwargs):
        """
        Initializes the BayesianNNRegressor object with given or default parameters.
        """
        super().__init__(**kwargs)

    def _init_model(self):
        """
        Initializes the Bayesian Neural Network layers of the model based on configuration.
        """
        if isinstance(cfg.prob.hidden_size, int):
            self.hidden_sizes = [cfg.prob.hidden_size]
        else:
            self.hidden_sizes = cfg.prob.hidden_size

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
        """
        Defines the forward pass of the Bayesian Neural Network.

        Args:
            x_data (torch.Tensor): The input data tensor.
            y_data (torch.Tensor): The target data tensor.

        Returns:
            torch.Tensor: The output tensor of the model.
        """
        x =  self.layers(x_data)
        df = pyro.sample("df", dist.Exponential(1.))
        scale = pyro.sample("scale", dist.HalfCauchy(2.5))
        with pyro.plate("data", x_data.shape[0]):
            obs = pyro.sample("obs", dist.StudentT(df, x, scale).to_event(1), 
                              obs=y_data)
            
        return x
    
    def _initSVI(self) -> pyro.infer.svi.SVI:
        """
        Initializes Stochastic Variational Inference (SVI) for Bayesian Inference with an AutoNormal guide.

        Returns:
            pyro.infer.svi.SVI: The SVI object.
        """
        self.guide = AutoNormal(self.forward)
        return SVI(model=self.forward,
                   guide=self.guide,
                   optim=self.optimizer, 
                   loss=TraceMeanField_ELBO())
    
    def _predict(self,
                test_dl : torch.utils.data.DataLoader
                ) -> torch.Tensor:
        """
        Predicts the output for the given test data.

        Args:
            test_dl (torch.utils.data.DataLoader): The test data loader.

        Returns:
            torch.Tensor: The predicted output tensor.
        """
        return self._predictNN(test_dl)
    

    