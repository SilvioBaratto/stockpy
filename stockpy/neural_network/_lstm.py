import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from ._base_model import BaseRegressorRNN
from ._base_model import BaseClassifierRNN
from ..config import Config as cfg

class LSTMRegressor(BaseRegressorRNN):
    """
    A class representing a Long Short-Term Memory (LSTM) model for stock prediction.

    The LSTM model is designed to capture long-range dependencies in time series data, such as stock prices.
    It consists of an LSTM layer followed by a series of fully connected layers and activation functions.

    :param input_size: The number of input features for the LSTM model.
    :type input_size: int
    :param hidden_size: The number of hidden units in each LSTM layer.
    :type hidden_size: intinput_size
    :param num_layers: The number of LSTM layers in the model.
    :type num_layers: int
    :param output_size: The number of output units for the LSTM model, corresponding to the predicted target variable(s).
    :type output_size: int
    :param dropout: The dropout percentage applied between layers for regularization, preventing overfitting.
    :type dropout: float
    :example:
        >>> from stockpy.neural_network import LSTM
        >>> lstm = LSTM()
    """    
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=input_size,  # input_size is the number of features
                            hidden_size=cfg.nn.hidden_size, 
                            num_layers=cfg.nn.num_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(cfg.nn.hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LSTM model.

        :param x: The input tensor.
        :type x: torch.Tensor

        :returns: The output tensor, corresponding to the predicted target variable(s).
        :rtype: torch.Tensor
        """
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(cfg.nn.num_layers, 
                                  batch_size, 
                                  cfg.nn.hidden_size)).to(cfg.training.device)
        c0 = Variable(torch.zeros(cfg.nn.num_layers, 
                                  batch_size, 
                                  cfg.nn.hidden_size)).to(cfg.training.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc(hn[0])   
        out = out.view(-1,self.output_size)

        return out

class LSTMClassifier(BaseClassifierRNN):
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size=input_size,  # input_size is the number of features
                            hidden_size=cfg.nn.hidden_size, 
                            num_layers=cfg.nn.num_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(cfg.nn.hidden_size, output_size)
    
    # Write the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LSTM model.

        :param x: The input tensor.
        :type x: torch.Tensor

        :returns: The output tensor, corresponding to the predicted target variable(s).
        :rtype: torch.Tensor
        """
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(cfg.nn.num_layers, 
                                  batch_size, 
                                  cfg.nn.hidden_size)).to(cfg.training.device)
        c0 = Variable(torch.zeros(cfg.nn.num_layers, 
                                  batch_size, 
                                  cfg.nn.hidden_size)).to(cfg.training.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc(hn[0]) 
        out = out.view(-1, self.output_size)  # Reshape the output tensor to match the expected dimensions

        return out
