import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataclasses import dataclass

@dataclass
class ModelArgs:
    input_size: int = 4
    hidden_size: int = 8
    output_size: int = 1
    num_layers: int = 2
    dropout: float = 0.2

class _MLP(nn.Module):
    """
    A class representing a neural network model for time series prediction.

    Parameters:
        input_size (int): the number of input features
        hidden_size (int): the number of hidden units in the GRU layer
        num_layers (int): the number of GRU layers
        output_dim (int): the number of output units
        dropout (float): dropout percentage
    """
    def __init__(self, 
                args: ModelArgs):
        
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(args.input_size, args.hidden_size),   # [4] -> [8]
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_size, args.hidden_size * 2), # [8] -> [16]
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_size * 2, args.hidden_size * 2), # [16] -> [16]
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_size * 2, args.hidden_size), # [16] -> [8]
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_size, args.output_size) # [8] -> [1]
        )

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters:
            x (torch.Tensor): the input tensor

        Returns:
            the output tensor
        """
        return self.layers(x)

    @property
    def model_type(self):
        return "neural_network"
