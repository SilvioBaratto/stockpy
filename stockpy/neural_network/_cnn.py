import os
import torch
import torch.nn as nn
from ._base_model import BaseRegressorCNN
from ._base_model import BaseClassifierCNN
import torch.nn.functional as F
from ..config import Config as cfg

class CNNRegressor(BaseRegressorCNN):
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

        self.layers = nn.Sequential(
            nn.Conv1d(1, cfg.nn.num_filters, cfg.nn.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(cfg.nn.pool_size),
            nn.Flatten(),
            nn.Linear(cfg.nn.num_filters * ((input_size - cfg.nn.kernel_size + 1) // cfg.nn.pool_size), cfg.nn.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.comm.dropout),
            nn.Linear(cfg.nn.hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class CNNClassifier(BaseClassifierCNN):
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

        self.layers = nn.Sequential(
            nn.Conv1d(1, cfg.nn.num_filters, cfg.nn.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(cfg.nn.pool_size),
            nn.Flatten(),
            nn.Linear(cfg.nn.num_filters * ((input_size - cfg.nn.kernel_size + 1) // cfg.nn.pool_size), cfg.nn.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.comm.dropout),
            nn.Linear(cfg.nn.hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)