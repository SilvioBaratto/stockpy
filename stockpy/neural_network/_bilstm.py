import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from ._base_model import BaseRegressorRNN
from ._base_model import BaseClassifierRNN
from ..config import Config as cfg

class BiLSTMRegressor(BaseRegressorRNN):
    """
    A class representing a Bidirectional Long Short-Term Memory (BiLSTM) model for stock prediction.

    The BiLSTM model extends the LSTM model by processing the input data in both forward and backward directions,
    allowing it to capture both past and future dependencies in time series data, such as stock prices.
    It consists of a bidirectional LSTM layer followed by a series of fully connected layers and activation functions.

    :param input_size: The number of input features for the BiLSTM model.
    :type input_size: int
    :param hidden_size: The number of hidden units in each LSTM layer.
    :type hidden_size: int
    :param num_layers: The number of LSTM layers in the model.
    :type num_layers: int
    :param output_size: The number of output units for the BiLSTM model, corresponding to the predicted target variable(s).
    :type output_size: int
    :param dropout: The dropout percentage applied between layers for regularization, preventing overfitting.
    :type dropout: float
    :example:
        >>> from stockpy.neural_network import BiLSTM
        >>> bilstm = BiLSTM()
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
                            batch_first=True,
                            bidirectional=True)
                
        self.fc = nn.Linear(cfg.nn.hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the BiLSTM model.

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
        c0 = Variable(torch.zeros(cfg.nn.num_layers * 2, 
                                  batch_size, 
                                  cfg.nn.hidden_size)
                                  ).to(cfg.training.device)
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc(hn[0])   
        out = out.view(-1,self.output_size)

        return out
        
class BiLSTMClassifier(BaseClassifierRNN):
    """
    A class representing a Bidirectional Long Short-Term Memory (BiLSTM) model for stock prediction.

    The BiLSTM model extends the LSTM model by processing the input data in both forward and backward directions,
    allowing it to capture both past and future dependencies in time series data, such as stock prices.
    It consists of a bidirectional LSTM layer followed by a series of fully connected layers and activation functions.

    :param input_size: The number of input features for the BiLSTM model.
    :type input_size: int
    :param hidden_size: The number of hidden units in each LSTM layer.
    :type hidden_size: int
    :param num_layers: The number of LSTM layers in the model.
    :type num_layers: int
    :param output_size: The number of output units for the BiLSTM model, corresponding to the predicted target variable(s).
    :type output_size: int
    :param dropout: The dropout percentage applied between layers for regularization, preventing overfitting.
    :type dropout: float
    :example:
        >>> from stockpy.neural_network import BiLSTM
        >>> bilstm = BiLSTM()
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
                            batch_first=True,
                            bidirectional=True)
                
        self.fc = nn.Linear(cfg.nn.hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the BiLSTM model.

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
        c0 = Variable(torch.zeros(cfg.nn.num_layers * 2, 
                                  batch_size, 
                                  cfg.nn.hidden_size)
                                  ).to(cfg.training.device)
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc(hn[0])   
        out = out.view(-1,self.output_size)

        return out