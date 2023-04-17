import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from ..config import prob_args, shared


class BayesianNN(PyroModule):
    """
    This class implements a Bayesian Neural Network model using Pyro. 
    It consists of three linear layers with ReLU activation and dropout applied between them. 
    The final layer uses a Student's t-distribution instead of a Normal distribution. 
    The model is trained using maximum likelihood estimation.

    :param input_size: the number of input features
    :type input_size: int
    :param hidden_size: the number of hidden units in the GRU layer
    :type hidden_size: int
    :param num_layers: the number of GRU layers
    :type num_layers: int
    :param output_dim: the number of output units
    :type output_dim: int

    :ivar layers: a PyroModule containing three linear layers with ReLU activation and dropout applied between them.
    :vartype layers: PyroModule[nn.Sequential]

    :example:
        >>> from stockpy.probabilistic import BayesianNN
        >>> bayesian_nn = BayesianNN()
    """
    
    def __init__(self):
        """
        Initializes the Bayesian Neural Network model.

        :param prob_args: a class containing the model hyperparameters
        :type prob_args: ModelArgs
        """
        
        super().__init__()
        self.layers = PyroModule[nn.Sequential](
            PyroModule[nn.Linear](prob_args.input_size, 
                                  prob_args.hidden_size), # [4] -> [8]
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](shared.dropout),
            PyroModule[nn.Linear](prob_args.hidden_size, 
                                  prob_args.input_size), # [8] -> [4]
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](shared.dropout),
            PyroModule[nn.Linear](prob_args.input_size, 
                                  prob_args.output_size), # [4] -> [1]
        )

    def forward(self, x_data: torch.Tensor, 
                y_data: torch.Tensor=None) -> torch.Tensor:
        """
        Computes the forward pass of the Bayesian Neural Network model using Pyro.

        :param x_data: the input data tensor
        :type x_data: torch.Tensor
        :param y_data: the target data tensor
        :type y_data: torch.Tensor

        :returns: the output tensor of the model
        :rtype: torch.Tensor
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
        """
        Returns the type of the model.

        :returns: the type of the model
        :rtype: str
        """
        return "probabilistic"
