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

class _BiLSTM(nn.Module):
    """
    A class representing a neural network model for time series prediction.

    Parameters:
        input_size (int): the number of input features
        hidden_size (int): the number of hidden units in the GRU layer
        num_layers (int): the number of GRU layers
        output_dim (int): the number of output units
    """
    def __init__(self, 
                args: ModelArgs):

        super().__init__()

        self.lstm = nn.LSTM(input_size=args.input_size, 
                          hidden_size=args.hidden_size * 2, 
                          num_layers=args.num_layers, 
                          batch_first=True,
                          bidirectional=True
                          )
        self.layers = nn.Sequential(
        nn.ReLU(),
        nn.Linear(args.hidden_size * 2 * 2, args.hidden_size * 2), # [32] -> [16]
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_size * 2, args.hidden_size * 2), # [16] -> [16]
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_size * 2, args.hidden_size), # [16] -> [8]
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_size, args.output_size), # [8] -> [1]
        )

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters:
            x (torch.Tensor): the input tensor

        Returns:
            out (torch.Tensor): the output tensor
        """
        
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(ModelArgs.num_layers*2, 
                                  batch_size, 
                                  ModelArgs.hidden_size * 2)
                                  )
        c0 = Variable(torch.zeros(ModelArgs.num_layers * 2, 
                                  batch_size, 
                                  ModelArgs.hidden_size * 2))
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.layers(out[:,-1,:])       
        out = out.view(-1,1)

        return out
    
    @property
    def model_type(self):
        return "neural_network"