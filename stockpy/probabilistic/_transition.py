import torch
import torch.nn as nn

class Transition(nn.Module):
    """
    Define Gaussian latent transition probabilities in sequential models.

    This module is used to model the evolution of the latent state `z_t` at time `t`,
    considering the previous latent state `z_{t-1}` and an external input `x_t`.

    Attributes
    ----------
    lin_gate_z_to_hidden : nn.Linear
        Learns the transformation from `z_{t-1}` to the gate's hidden layer.
    lin_gate_x_to_hidden : nn.Linear
        Learns the transformation from `x_t` to the gate's hidden layer.
    lin_gate_hidden_to_z : nn.Linear
        Learns the transformation from the gate's hidden layer to the gate values.
    lin_proposed_mean_z_to_hidden : nn.Linear
        Learns the transformation from `z_{t-1}` to the proposed mean's hidden layer.
    lin_proposed_mean_x_to_hidden : nn.Linear
        Learns the transformation from `x_t` to the proposed mean's hidden layer.
    lin_proposed_mean_hidden_to_z : nn.Linear
        Learns the transformation from the proposed mean's hidden layer to the mean values.
    lin_sig : nn.Linear
        Computes the scale parameter.
    lin_z_to_loc : nn.Linear
        Directly transforms `z_{t-1}` to the location parameter.
    lin_x_to_loc : nn.Linear
        Directly transforms `x_t` to the location parameter.
    relu : nn.ReLU
        Applies a rectified linear unit activation function.
    softplus : nn.Softplus
        Applies the softplus function to ensure the scale parameter is positive.

    Parameters
    ----------
    z_dim : int
        The size of the latent state `z_t`.
    input_dim : int
        The size of the input `x_t`.
    transition_dim : int
        The size of the hidden layer for the transformations.

    Notes
    -----
    The transition dynamics are defined by a Gaussian distribution with a mean (location) and
    scale (variance) that are computed through learned transformations involving the current input
    and the previous latent state.
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

    def forward(self, z_t_1: torch.Tensor, x_t: torch.Tensor):
        """
        The forward pass for the `Transition` module computes the mean (`loc`) and scale 
        for the Gaussian distribution governing the transition of latent states. This 
        includes a gating mechanism to blend between the previous state and a proposed 
        new state based on the input.

        Parameters
        ----------
        z_t_1 : torch.Tensor
            The latent state at time `t-1`.
        x_t : torch.Tensor
            The input at time `t`.

        Returns
        -------
        tuple
            A tuple containing the location (`loc`) and scale parameters for the Gaussian
            distribution of `z_t`.
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