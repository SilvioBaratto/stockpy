import torch
import torch.nn as nn
import torch.nn.functional as F

from stockpy.utils import get_activation_function

class Combiner(nn.Module):
    """
    The `Combiner` class parameterizes the variational distribution `q(z_t | z_{t-1}, x_{t:T})`,
    where `z_t` represents the latent variable at time `t`, and `x_{t:T}` is the sequence
    of observations from time `t` to `T`. It serves as a building block of the guide in variational
    inference, especially within the context of sequential or time-series models.

    Attributes:
        lin_z_to_hidden (nn.Linear): A linear transformation that maps the latent space
            at `t-1` to the hidden state dimension.
        lin_hidden_to_loc (nn.Linear): A linear transformation that maps the hidden state
            to the location (mean) parameter of the latent space at `t`.
        lin_hidden_to_scale (nn.Linear): A linear transformation that maps the hidden state
            to the scale (standard deviation) parameter of the latent space at `t`.
        tanh (nn.Tanh): The hyperbolic tangent non-linearity.
        softplus (nn.Softplus): The softplus non-linearity, ensuring the scale parameter is positive.

    Args:
        z_dim (int): The dimensionality of the latent space `z`.
        rnn_dim (int): The dimensionality of the RNN's hidden state.
    """

    def __init__(self, z_dim: int, rnn_dim: int):
        super().__init__()
        # Initialize the three linear transformations used in the neural network.
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        
        # Initialize the two non-linearities used in the neural network.
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1: torch.Tensor, h_rnn: torch.Tensor):
        """
        Defines the forward pass of the Combiner.

        Given the latent variable `z` at time `t-1` and the hidden state `h_rnn` of an RNN
        conditioned on observations `x_{t:T}`, this method computes the parameters (mean and scale)
        of the Gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`.

        Args:
            z_t_1 (torch.Tensor): The latent variable at time `t-1`.
            h_rnn (torch.Tensor): The hidden state of the RNN which encodes `x_{t:T}`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the location (mean) and
            scale (standard deviation) parameters of the variational distribution for the latent
            variable `z_t`.
        """
        # Process the input latent variable through a non-linearity after linear transformation.
        h_latent = self.tanh(self.lin_z_to_hidden(z_t_1))
        
        # Combine the rnn hidden state with a transformed version of z_t_1.
        h_combined = 0.5 * (h_latent + h_rnn)
        
        # Compute the location (mean) parameter for the Gaussian distribution of z_t.
        loc = self.lin_hidden_to_loc(h_combined)
        
        # Compute the scale (standard deviation) parameter for the Gaussian distribution of z_t.
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        
        # Return the mean and standard deviation as parameters of the Gaussian distribution.
        return loc, scale