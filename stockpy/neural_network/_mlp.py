import os
import torch
import torch.nn as nn
from ._base_model import BaseRegressorFFNN
from ._base_model import BaseClassifierFFNN
from ..config import Config as cfg

class MLPRegressor(BaseRegressorFFNN):
    """
    A class representing a Multilayer Perceptron (MLP) neural network model for stock prediction.

    The MLP model consists of a series of linear layers, each followed by a non-linear activation function,
    such as ReLU, and dropout for regularization. The model is designed to learn non-linear relationships
    between input features and the target variable for predicting stock prices.

    :param input_size: The number of input features for the MLP model.
    :type input_size: int
    :param hidden_size: The number of hidden units in each hidden layer of the MLP model.
    :type hidden_size: int
    :param num_layers: The number of layers in the MLP model, including input, hidden, and output layers.
    :type num_layers: int
    :param output_dim: The number of output units for the MLP model, corresponding to the predicted target variable(s).
    :type output_dim: int
    :param dropout: The dropout percentage applied between layers for regularization, preventing overfitting.
    :type dropout: float
    :example:
        >>> from stockpy.neural_network import MLP
        >>> mlp = MLP()
    """
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        """
        Initializes the MLP neural network model.

        :param args: The arguments to configure the model.
        :type args: ModelArgs
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, cfg.nn.hidden_size),   # [input_size] -> [hidden_size]
            nn.ReLU(),
            nn.Dropout(cfg.comm.dropout),
            nn.Linear(cfg.nn.hidden_size, cfg.nn.hidden_size),  # [hidden_size] -> [hidden_size]
            nn.ReLU(),
            nn.Dropout(cfg.comm.dropout),
            nn.Linear(cfg.nn.hidden_size, input_size),   # [hidden_size] -> [input_size]
            nn.ReLU(),
            nn.Dropout(cfg.comm.dropout),
            nn.Linear(input_size, output_size),   # [input_size] -> [output_size]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        :param x: The input tensor.
        :type x: torch.Tensor

        :returns: The output tensor, corresponding to the predicted target variable(s).
        :rtype: torch.Tensor
        """
        return self.layers(x)
    
class MLPClassifier(BaseClassifierFFNN):
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        """
        Initializes the MLP neural network model.

        :param args: The arguments to configure the model.
        :type args: ModelArgs
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, cfg.nn.hidden_size),   # [input_size] -> [hidden_size]
            nn.ReLU(),
            nn.Dropout(cfg.comm.dropout),
            nn.Linear(cfg.nn.hidden_size, cfg.nn.hidden_size),  # [hidden_size] -> [hidden_size]
            nn.ReLU(),
            nn.Dropout(cfg.comm.dropout),
            nn.Linear(cfg.nn.hidden_size, input_size),   # [hidden_size] -> [input_size]
            nn.ReLU(),
            nn.Dropout(cfg.comm.dropout),
            nn.Linear(input_size, output_size),   # [input_size] -> [output_size]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        :param x: The input tensor.
        :type x: torch.Tensor

        :returns: The output tensor, corresponding to the predicted target variable(s).
        :rtype: torch.Tensor
        """
        return self.layers(x)