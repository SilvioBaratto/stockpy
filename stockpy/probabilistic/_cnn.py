import os
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from ._base_model import BaseRegressorCNN
from ._base_model import BaseClassifierCNN
from ..config import Config as cfg

class BayesianCNNRegressor(BaseRegressorCNN):
    """
    A class representing a Convolutional Neural Network (CNN) model for stock prediction.

    The CNN model consists of a series of convolutional layers, each followed by a non-linear activation function,
    such as ReLU, and pooling layers to downsample the feature maps. The model is designed to learn non-linear relationships
    between input features and the target variable for predicting stock prices.

    :param input_size: The number of input features for the CNN model.
    :type input_size: int
    :param num_filters: The number of filters in the convolutional layer of the CNN model.
    :type num_filters: int
    :param filter_size: The size of the filters in the convolutional layer of the CNN model.
    :type filter_size: int
    :param output_dim: The number of output units for the CNN model, corresponding to the predicted target variable(s).
    :type output_dim: int
    :param dropout: The dropout percentage applied after the convolutional layer for regularization, preventing overfitting.
    :type dropout: float
    :example:
        >>> from stockpy.neural_network import CNN
        >>> cnn = CNN()
    """
    def __init__(self):
        super().__init__()
        self.layers = PyroModule[nn.Sequential](
            PyroModule[nn.Conv1d](1, cfg.nn.num_filters, cfg.nn.kernel_size),
            PyroModule[nn.ReLU](),
            PyroModule[nn.MaxPool1d](cfg.nn.pool_size),
            PyroModule[nn.Flatten](),
            PyroModule[nn.Linear](cfg.nn.num_filters * (cfg.nn.input_size - cfg.nn.kernel_size + 1) // cfg.nn.pool_size, cfg.nn.hidden_size),
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](cfg.shared.dropout),
            PyroModule[nn.Linear](cfg.nn.hidden_size, cfg.nn.output_size)
        )

    def forward(self, 
                x_data: torch.Tensor,
                y_data: torch.Tensor=None) -> torch.Tensor:
        x = self.layers(x_data)
        # use StudentT distribution instead of Normal
        df = pyro.sample("df", dist.Exponential(1.))
        scale = pyro.sample("scale", dist.HalfCauchy(2.5))
        with pyro.plate("data", x_data.shape[0]):
            obs = pyro.sample("obs", dist.StudentT(df, x, scale).to_event(1), 
                              obs=y_data)
            
        return x