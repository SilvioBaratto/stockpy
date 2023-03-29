import sys
sys.path.append('../')
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from dataclasses import dataclass

@dataclass
class ModelArgs:
    input_size: int = 4
    hidden_size: int = 8
    output_size: int = 1
    num_layers: int = 2
    dropout: float = 0.2


class _BayesianNN(PyroModule):
    """
    This class implements a Bayesian Neural Network model using Pyro. 
    It consists of three linear layers with ReLU activation and dropout applied between them. 
    The final layer uses a Student's t-distribution instead of a Normal distribution. 
    The model is trained using maximum likelihood estimation

    Parameters:
        input_size (int): the number of input features
        hidden_size (int): the number of hidden units in the GRU layer
        num_layers (int): the number of GRU layers
        output_dim (int): the number of output units
    """

    def __init__(self,
                 args: ModelArgs):
        
        super().__init__()
        self.layers = PyroModule[nn.Sequential](
            PyroModule[nn.Linear](args.input_size, 
                                  args.hidden_size), # [4] -> [8]
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](args.dropout),
            PyroModule[nn.Linear](args.hidden_size, 
                                  args.hidden_size * 2), # [8] -> [16]
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](args.dropout),
            PyroModule[nn.Linear](args.hidden_size * 2, 
                                  args.hidden_size * 2), # [16] -> [16]
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](args.dropout),
            PyroModule[nn.Linear](args.hidden_size * 2, 
                                  args.hidden_size), # [16] -> [8]
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](args.dropout),
            PyroModule[nn.Linear](args.hidden_size, 
                                  args.output_size), # [8] -> [1]
        )

    def forward(self, x_data, y_data=None):
        """
        This function computes the forward pass of the Bayesian Neural Network model using Pyro.
        
        Args:
            x_data (torch.Tensor): the input data tensor
            y_data (torch.Tensor): the target data tensor
            
        Returns:
            torch.Tensor: the output tensor of the model
        """
        x =  self.layers(x_data)
        # use StudentT distribution instead of Normal
        df = pyro.sample("df", dist.Exponential(1.))
        scale = pyro.sample("scale", dist.HalfCauchy(2.5))
        with pyro.plate("data", x_data.shape[0]):
            obs = pyro.sample("obs", dist.StudentT(df, x, scale).to_event(1), 
                              obs=y_data)
            
        return x
    
    @property
    def model_type(self):
        return "probabilistic"
