import os
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from ._base_model import BaseRegressorCNN
from ._base_model import BaseClassifierCNN
from ..config import Config as cfg

class PyroLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

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
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.layers = PyroModule[nn.Sequential](
            PyroModule[nn.Conv1d](1, cfg.prob.num_filters, cfg.prob.kernel_size),
            PyroModule[nn.ReLU](),
            PyroModule[nn.MaxPool1d](cfg.prob.pool_size),
            PyroModule[nn.Flatten](),
            PyroModule[nn.Linear](cfg.prob.num_filters * (input_size - cfg.prob.kernel_size + 1) // cfg.prob.pool_size, cfg.prob.hidden_size),
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](cfg.comm.dropout),
            PyroModule[nn.Linear](cfg.prob.hidden_size, output_size)
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
    
class BayesianCNNClassifier(BaseClassifierCNN):
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
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.layers = PyroModule[nn.Sequential](
            PyroModule[nn.Conv1d](1, cfg.prob.num_filters, cfg.prob.kernel_size),
            PyroModule[nn.ReLU](),
            PyroModule[nn.MaxPool1d](cfg.prob.pool_size),
            PyroModule[nn.Flatten](),
            PyroModule[PyroLinear](cfg.prob.num_filters * (input_size - cfg.prob.kernel_size + 1) // cfg.prob.pool_size, cfg.prob.hidden_size),
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](cfg.comm.dropout),
            PyroModule[PyroLinear](cfg.prob.hidden_size, output_size),
            PyroModule[nn.Softmax](dim=1)
        )

    def forward(self, 
                x_data: torch.Tensor,
                y_data: torch.Tensor=None) -> torch.Tensor:
        
        x = self.layers(x_data)
        with pyro.plate("data", x_data.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(probs=x).to_event(1), obs=y_data)
            
        return x