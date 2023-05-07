import os
import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.nn.functional as F
from ..config import Config as cfg

class Emitter(nn.Module):
    """
    Parameterizes the Gaussian observation likelihood p(y_t | z_t, x_t).

    :ivar lin_z_to_hidden: a linear transformation from the latent space to a hidden state
    :vartype lin_z_to_hidden: torch.nn.Linear
    :ivar lin_x_to_hidden: a linear transformation from the input space to a hidden state
    :vartype lin_x_to_hidden: torch.nn.Linear
    :ivar lin_hidden_to_mean: a linear transformation from the hidden state to the output mean
    :vartype lin_hidden_to_mean: torch.nn.Linear
    :ivar variance: the fixed variance hyperparameter
    :vartype variance: torch.nn.Parameter
    :ivar relu: a ReLU activation function
    :vartype relu: torch.nn.ReLU

    :example:
        >>> emitter = Emitter()
        >>> print(emitter)
        Emitter(
          (lin_z_to_hidden): Linear(in_features=16, out_features=32, bias=True)
          (lin_x_to_hidden): Linear(in_features=4, out_features=32, bias=True)
          (lin_hidden_to_mean): Linear(in_features=32, out_features=1, bias=True)
          (relu): ReLU()
        )
    """

    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(cfg.prob.z_dim, cfg.prob.emission_dim)
        self.lin_x_to_hidden = nn.Linear(input_size, cfg.prob.emission_dim)
        self.lin_hidden_to_mean = nn.Linear(cfg.prob.emission_dim, 1)
        # initialize the fixed variance hyperparameter
        self.variance = nn.Parameter(torch.tensor(cfg.prob.variance))
        # initialize the non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, 
                z_t: torch.Tensor, 
                x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.nn.parameter.Parameter]:
        """
        Given the latent z at a particular time step t and the input x at time step t,
        we return the mean and variance of the Gaussian distribution p(y_t|z_t, x_t).

        :param z_t: the latent variable at time step t
        :type z_t: torch.Tensor
        :param x_t: the input variable at time step t
        :type x_t: torch.Tensor
        :return: a tuple containing the mean and variance of the Gaussian distribution
        :rtype: tuple
        """

        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_x_to_hidden(x_t))
        h3 = h1 + h2  # element-wise sum of the two hidden states
        mean = self.lin_hidden_to_mean(h3)
        return mean, self.variance
    

class GatedTransition(nn.Module):
    """
    Parameterizes the dynamics of the latent variables z_t.

    :ivar lin_z_to_hidden: a linear transformation from the latent space to a hidden state
    :vartype lin_z_to_hidden: torch.nn.Linear
    :ivar lin_x_to_hidden: a linear transformation from the input space to a hidden state
    :vartype lin_x_to_hidden: torch.nn.Linear
    :ivar lin_hidden_to_hidden1: the first gated transformation
    :vartype lin_hidden_to_hidden1: torch.nn.Linear
    :ivar lin_hidden_to_hidden2: the second gated transformation
    :vartype lin_hidden_to_hidden2: torch.nn.Linear
    :ivar relu: a ReLU activation function
    :vartype relu: torch.nn.ReLU
    :ivar sigmoid: a sigmoid activation function
    :vartype sigmoid: torch.nn.Sigmoid

    :example:
        >>> gated_transition = GatedTransition()
        >>> print(gated_transition)
        GatedTransition(
          (lin_z_to_hidden): Linear(in_features=16, out_features=32, bias=True)
          (lin_x_to_hidden): Linear(in_features=4, out_features=32, bias=True)
          (lin_hidden_to_hidden1): Linear(in_features=32, out_features=32, bias=True)
          (lin_hidden_to_hidden2): Linear(in_features=32, out_features=32, bias=True)
          (relu): ReLU()
          (sigmoid): Sigmoid()
        )
    """
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        super().__init__()
        # initialize the two linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(cfg.prob.z_dim, cfg.prob.transition_dim)
        self.lin_x_to_hidden = nn.Linear(input_size, cfg.prob.transition_dim)
        # initialize the two gated transformations used in the neural network
        self.lin_hidden_to_hidden1 = nn.Linear(cfg.prob.transition_dim, cfg.prob.transition_dim)
        self.lin_hidden_to_hidden2 = nn.Linear(cfg.prob.transition_dim, cfg.prob.transition_dim)
        # initialize the non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, 
                z_t: torch.Tensor, 
                x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.nn.parameter.Parameter]:
        """
        Given the latent z at a particular time step t and the input x at time step t,
        we return the parameters for the Gaussian distribution p(z_t | z_{t-1}, x_t).

        :param z_t: the latent variable at time step t
        :type z_t: torch.Tensor
        :param x_t: the input variable at time step t
        :type x_t: torch.Tensor
        :return: a tuple containing the mean and variance of the Gaussian distribution
        :rtype: tuple
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_x_to_hidden(x_t))
        gated1 = self.sigmoid(self.lin_hidden_to_hidden1(h1))
        gated2 = self.sigmoid(self.lin_hidden_to_hidden2(h2))
        h3 = gated1 * h1 + gated2 * h2
        mu_t = h3
        return mu_t, 1.0

    
class Combiner(nn.Module):
    """
    Combines the previous hidden state z_{t-1} and the current input x_t
    to produce the hidden state h_t, which is used by the emitter and
    transition networks.

    :ivar lin_z_to_hidden: a linear transformation from the latent space to a hidden state
    :vartype lin_z_to_hidden: torch.nn.Linear
    :ivar lin_rnn_to_hidden: a linear transformation from the RNN output to a hidden state
    :vartype lin_rnn_to_hidden: torch.nn.Linear
    :ivar hidden_to_loc: a linear transformation from the hidden state to the mean of the Gaussian distribution
    :vartype hidden_to_loc: torch.nn.Linear
    :ivar hidden_to_scale: a linear transformation from the hidden state to the pre-activation scale of the Gaussian distribution
    :vartype hidden_to_scale: torch.nn.Linear
    """
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(cfg.prob.z_dim, cfg.prob.emission_dim)
        self.lin_rnn_to_hidden = nn.Linear(cfg.prob.rnn_dim, cfg.prob.emission_dim)
        self.hidden_to_loc = nn.Linear(cfg.prob.emission_dim, cfg.prob.z_dim)
        self.hidden_to_scale = nn.Linear(cfg.prob.emission_dim, cfg.prob.z_dim)
        self.relu = nn.ReLU()

    def forward(self, 
                z_prev : torch.Tensor, 
                rnn_input : torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combines the previous hidden state z_{t-1} and the current input x_t
        to produce the hidden state h_t, which is used by the emitter and
        transition networks.

        :param z_prev: the previous latent variable
        :type z_prev: torch.Tensor
        :param rnn_input: the current input variable
        :type rnn_input: torch.Tensor
        :return: a tuple containing the mean and variance of the Gaussian distribution
        :rtype: tuple
        """
        hidden = self.relu(self.lin_z_to_hidden(z_prev) + self.lin_rnn_to_hidden(rnn_input))
        loc = self.hidden_to_loc(hidden)
        scale = F.softplus(self.hidden_to_scale(hidden))
        return loc, scale
    
