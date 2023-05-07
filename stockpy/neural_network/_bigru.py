import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from ._base_model import BaseRegressorRNN
from ._base_model import BaseClassifierRNN
from ..config import Config as cfg

class BiGRURegressor(BaseRegressorRNN):
    """
    A class representing a Bidirectional Gated Recurrent Unit (BiGRU) model for stock prediction.

    The BiGRU model is a type of recurrent neural network (RNN) that is capable of learning long-term dependencies
    in time series data, such as stock prices. The model consists of a bidirectional GRU layer followed by a series of
    fully connected layers and activation functions. Bidirectional GRUs process the input sequence both forward and
    backward, allowing the model to capture information from both the past and the future.

    :param input_size: The number of input features for the BiGRU model.
    :type input_size: int
    :param hidden_size: The number of hidden units in each GRU layer.
    :type hidden_size: int
    :param num_layers: The number of GRU layers in the model.
    :type num_layers: int
    :param output_size: The number of output units for the BiGRU model, corresponding to the predicted target variable(s).
    :type output_size: int
    :param dropout: The dropout percentage applied between layers for regularization, preventing overfitting.
    :type dropout: float
    :example:
        >>> from stockpy.neural_network import BiGRU
        >>> bigru = BiGRU()
    """
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=cfg.nn.hidden_size, 
                          num_layers=cfg.nn.num_layers, 
                          batch_first=True,
                          bidirectional=True
                          )
        self.fc = nn.Linear(cfg.nn.hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the BiGRU model.
        :param x: The input tensor.
        :type x: torch.Tensor
        :returns: The output tensor, corresponding to the predicted target variable(s).
        :rtype: torch.Tensor
        """
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(cfg.nn.num_layers * 2, 
                                  batch_size, 
                                  cfg.nn.hidden_size)
                                  ).to(cfg.training.device)
        
        _, (hn) = self.gru(x, (h0))
        out = self.fc(hn[0])     
        out = out.view(-1, self.output_size)

        return out
        
class BiGRUClassifier(BaseClassifierRNN):
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=cfg.nn.hidden_size, 
                          num_layers=cfg.nn.num_layers, 
                          batch_first=True,
                          bidirectional=True
                          )
        self.fc = nn.Linear(cfg.nn.hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the BiGRU model.
        :param x: The input tensor.
        :type x: torch.Tensor
        :returns: The output tensor, corresponding to the predicted target variable(s).
        :rtype: torch.Tensor
        """
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(cfg.nn.num_layers * 2, 
                                  batch_size, 
                                  cfg.nn.hidden_size)
                                  ).to(cfg.training.device)
        
        _, (hn) = self.gru(x, (h0))
        out = self.fc(hn[0])     
        out = out.view(-1, self.output_size)

        return out