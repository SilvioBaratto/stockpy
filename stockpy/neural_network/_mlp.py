import torch
import torch.nn as nn
from torch.autograd import Variable
from ..config import ModelArgs as args

class _MLP(nn.Module):
    """
    A class representing a neural network model for stock prediction.

    :param input_size: the number of input features
    :type input_size: int
    :param hidden_size: the number of hidden units in the GRU layer
    :type hidden_size: int
    :param num_layers: the number of GRU layers
    :type num_layers: int
    :param output_dim: the number of output units
    :type output_dim: int
    :param dropout: dropout percentage
    :type dropout: float
    """
    def __init__(self):
        """
        Initializes the MLP neural network model.

        :param args: the arguments to configure the model
        :type args: ModelArgs
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.input_size, args.hidden_size),   # [4] -> [8]
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, args.hidden_size), # [8] -> [8]
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

        :returns: the output tensor
        :rtype: torch.Tensor
        """
        return self.layers(x)

    @property
    def model_type(self):
        """
        Returns the type of model.

        :returns: the model type
        :rtype: str
        """
        return "neural_network"