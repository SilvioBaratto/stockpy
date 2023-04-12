import torch
import torch.nn as nn
from torch.autograd import Variable
from ..config import nn_args, shared

class LSTM(nn.Module):
    """
    A class representing a Long Short-Term Memory (LSTM) model for stock prediction.

    The LSTM model is designed to capture long-range dependencies in time series data, such as stock prices.
    It consists of an LSTM layer followed by a series of fully connected layers and activation functions.

    :param input_size: The number of input features for the LSTM model.
    :type input_size: int
    :param hidden_size: The number of hidden units in each LSTM layer.
    :type hidden_size: int
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
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=nn_args.input_size, 
                            hidden_size=nn_args.hidden_size * 2, 
                            num_layers=nn_args.num_layers, 
                            batch_first=True)
        
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(nn_args.hidden_size * 2, nn_args.hidden_size), # [2 * hidden_size] -> [hidden_size]
            nn.ReLU(),
            nn.Dropout(shared.dropout),
            nn.Linear(nn_args.hidden_size, nn_args.input_size), # [hidden_size] -> [input_size]
            nn.ReLU(),
            nn.Dropout(shared.dropout),
            nn.Linear(nn_args.input_size, nn_args.output_size), # [input_size] -> [output_size]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LSTM model.

        :param x: The input tensor.
        :type x: torch.Tensor

        :returns: The output tensor, corresponding to the predicted target variable(s).
        :rtype: torch.Tensor
        """
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(nn_args.num_layers, batch_size, nn_args.hidden_size * 2))
        c0 = Variable(torch.zeros(nn_args.num_layers, batch_size, nn_args.hidden_size * 2))
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.layers(hn[0])       
        out = out.view(-1,1)

        return out
    
    @property
    def model_type(self) -> str:
        """
        Returns the type of model.

        :returns: The model type as a string.
        :rtype: str
        """
        return "neural_network"