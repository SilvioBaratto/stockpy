import torch
import torch.nn as nn
import torch.nn.functional as F
from stockpy.utils import get_activation_function

class Transition(nn.Module):
    """
    The `Transition` module defines the Gaussian latent transition probability 
    `p(z_t | z_{t-1}, x_t)`. This probability models the evolution of the latent
    state `z_t` at time `t` given the previous latent state `z_{t-1}` and an additional
    input `x_t`. It is used in sequential models where the state transitions are 
    probabilistic and potentially influenced by external inputs.

    The model combines learned gates and proposed means to compute the mean (`loc`) 
    and scale of the Gaussian distribution that defines the transition dynamics.

    Attributes:
        lin_gate_z_to_hidden (nn.Linear): Transformation from `z_{t-1}` to gate's hidden layer.
        lin_gate_x_to_hidden (nn.Linear): Transformation from `x_t` to gate's hidden layer.
        lin_gate_hidden_to_z (nn.Linear): Transformation from gate's hidden layer to gate values.
        lin_proposed_mean_z_to_hidden (nn.Linear): Transformation from `z_{t-1}` to proposed mean's hidden layer.
        lin_proposed_mean_x_to_hidden (nn.Linear): Transformation from `x_t` to proposed mean's hidden layer.
        lin_proposed_mean_hidden_to_z (nn.Linear): Transformation from proposed mean's hidden layer to mean values.
        lin_sig (nn.Linear): Transformation for computing the scale parameter.
        lin_z_to_loc (nn.Linear): Direct transformation from `z_{t-1}` to location parameter.
        lin_x_to_loc (nn.Linear): Direct transformation from `x_t` to location parameter.
        relu (nn.ReLU): Activation function used for non-linearity.
        softplus (nn.Softplus): Activation function to ensure the scale is positive.
    
    Args:
        z_dim (int): Dimensionality of the latent state `z_t`.
        input_dim (int): Dimensionality of the input `x_t`.
        transition_dim (int): Dimensionality of the hidden layer for transformation.
    """

    def __init__(self, z_dim: int, input_dim: int, transition_dim: int):
        super().__init__()
        # Initialize layers for gate mechanism
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_x_to_hidden = nn.Linear(input_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)

        # Initialize layers for proposing mean
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_x_to_hidden = nn.Linear(input_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)

        # Initialize layer to compute scale directly combining `z` and `x`
        self.lin_sig = nn.Linear(z_dim + input_dim, z_dim)

        # Initialize direct transformation layers for `z` and `x` to `loc`
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_x_to_loc = nn.Linear(input_dim, z_dim)

        # Enforcing identity transformation for `z` to `loc` initially
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        # Non-linear activation functions
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1: torch.Tensor, x_t: torch.Tensor) -> tuple:
        """
        The forward pass for the `Transition` module computes the mean (`loc`) and scale 
        for the Gaussian distribution governing the transition of latent states. This 
        includes a gating mechanism to blend between the previous state and a proposed 
        new state based on the input.

        Args:
            z_t_1 (torch.Tensor): The latent state at time `t-1`.
            x_t (torch.Tensor): The input at time `t`.

        Returns:
            tuple: A tuple containing the location (`loc`) and scale parameters for the 
                   Gaussian distribution of `z_t`.
        """
        # Gate computation
        _gate_z = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        _gate_x = self.relu(self.lin_gate_x_to_hidden(x_t))
        _gate = _gate_z + _gate_x
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))

        # Proposed mean computation
        _proposed_mean_z = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        _proposed_mean_x = self.relu(self.lin_proposed_mean_x_to_hidden(x_t))
        _proposed_mean = _proposed_mean_z + _proposed_mean_x
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)

        # Location computation
        loc_z = self.lin_z_to_loc(z_t_1)
        loc_x = self.lin_x_to_loc(x_t)
        loc = (1 - gate) * loc_z + gate * (loc_z + loc_x)

        # Scale computation
        combined_for_scale = torch.cat((self.relu(proposed_mean), x_t), dim=1)
        scale = self.softplus(self.lin_sig(combined_for_scale))

        return loc, scale





