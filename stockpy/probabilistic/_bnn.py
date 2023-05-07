import os
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from ._base_model import BaseRegressorFFNN
from ._base_model import BaseClassifierFFNN
from ..config import Config as cfg

class PyroLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

class BayesianNNRegressor(BaseRegressorFFNN):
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
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        """
        Initializes the Bayesian Neural Network model.

        :param prob_args: a class containing the model hyperparameters
        :type prob_args: ModelArgs
        """
        
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.layers = PyroModule[nn.Sequential](
            PyroModule[nn.Linear](input_size, 
                                  cfg.prob.hidden_size), # [4] -> [8]
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](cfg.comm.dropout),
            PyroModule[nn.Linear](cfg.prob.hidden_size, 
                                  input_size), # [8] -> [4]
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](cfg.comm.dropout),
            PyroModule[nn.Linear](input_size, 
                                  output_size), # [4] -> [1]
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
    
class BayesianNNClassifier(BaseRegressorFFNN):
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
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.layers = PyroModule[nn.Sequential](
            PyroModule[PyroLinear](input_size, cfg.nn.hidden_size),
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](cfg.comm.dropout),
            PyroModule[PyroLinear](cfg.nn.hidden_size, cfg.nn.hidden_size),
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](cfg.comm.dropout),
            PyroModule[PyroLinear](cfg.nn.hidden_size, input_size),
            PyroModule[nn.ReLU](),
            PyroModule[nn.Dropout](cfg.comm.dropout),
            PyroModule[PyroLinear](input_size, output_size),
            PyroModule[nn.Softmax](dim=1)
        )

    def forward(self, x_data: torch.Tensor, 
                y_data: torch.Tensor = None) -> torch.Tensor:
        
        x = self.layers(x_data)
        with pyro.plate("data", x_data.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(probs=x).to_event(1), obs=y_data)
            
        return x