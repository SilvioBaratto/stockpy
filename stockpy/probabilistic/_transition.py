import torch
import torch.nn as nn
import torch.nn.functional as F
from stockpy.utils import get_activation_function

class Transition(nn.Module):
    """
    Parameterizes the Gaussian latent transition probability `p(z_t | z_{t-1}, x_t)`
    """

    def __init__(self, z_dim, input_dim, transition_dim):
        super().__init__()
        
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_x_to_hidden = nn.Linear(input_dim, transition_dim)  # New line for x_t
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)

        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_x_to_hidden = nn.Linear(input_dim, transition_dim)  # New line for x_t
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)

        self.lin_sig = nn.Linear(z_dim + input_dim, z_dim)  # Updated to include x_dim

        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_x_to_loc = nn.Linear(input_dim, z_dim)  # New line for x_t

        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, x_t):
        """
        Given the latents `z_{t-1}` and inputs `x_t` at time t,
        return the mean and scale vectors that parameterize the
        (diagonal) Gaussian distribution `p(z_t | z_{t-1}, x_t)`
        """

        _gate_z = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        _gate_x = self.relu(self.lin_gate_x_to_hidden(x_t))  # New line for x_t
        _gate = _gate_z + _gate_x  # Combine both contributions
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))

        _proposed_mean_z = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        _proposed_mean_x = self.relu(self.lin_proposed_mean_x_to_hidden(x_t))  # New line for x_t
        _proposed_mean = _proposed_mean_z + _proposed_mean_x  # Combine both contributions
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)

        loc_z = self.lin_z_to_loc(z_t_1)
        loc_x = self.lin_x_to_loc(x_t)  # New line for x_t
        loc = (1 - gate) * loc_z + gate * (loc_z + loc_x)  # Updated to include x_t

        combined_for_scale = torch.cat((self.relu(proposed_mean), x_t), dim=1)  # Updated to include x_t
        scale = self.softplus(self.lin_sig(combined_for_scale))

        return loc, scale





