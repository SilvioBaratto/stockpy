import torch
import torch.nn as nn
from torch.autograd import Variable
from ..config import ModelArgs as args

class _GRU(nn.Module):
    """
    A class representing a GRJ model for stock prediction.

    :param input_size: the number of input features
    :type input_size: int
    :param hidden_size: the number of hidden units in the GRU layer
    :type hidden_size: int
    :param num_layers: the number of GRU layers
    :type num_layers: int
    :param output_size: the number of output units
    :type output_size: int
    :param dropout: dropout percentage
    :type dropout: float
    """
    def __init__(self):

        super().__init__()       
        self.gru = nn.GRU(input_size=args.input_size, 
                          hidden_size=args.hidden_size * 2, 
                          num_layers=args.num_layers, 
                          batch_first=True,
                          )
        
        self.layers = nn.Sequential(
        nn.ReLU(),
        nn.Linear(args.hidden_size * 2, args.hidden_size), # [16] -> [8]
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_size, args.input_size), # [8] -> [4]
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.input_size, args.output_size), # [4] -> [1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        :param x: the input tensor
        :type x: torch.Tensor
        :return: the output tensor
        :rtype: torch.Tensor
        """
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(args.num_layers, 
                                  batch_size, 
                                  args.hidden_size * 2))
        
        _, (hn) = self.gru(x, (h0))
        out = self.layers(hn[0])       
        out = out.view(-1,1)

        return out
    
    @property
    def model_type(self):
        return "neural_network"
