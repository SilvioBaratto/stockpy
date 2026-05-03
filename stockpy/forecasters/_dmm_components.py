"""Sub-modules used by ``DMMForecaster``.

Encapsulates the inference (``Combiner``) and generative (``Transition``,
``EmitterRegressor``) sub-networks of the Deep Markov Model. Each class is a
plain ``nn.Module`` and is unit-testable in isolation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Combiner", "EmitterRegressor", "Transition"]


class Combiner(nn.Module):
    """
    Parameterizes the variational distribution q(z_t | z_{t-1}, x_{t:T}).

    Maps the input and previous latent state to the parameters of the current
    latent state's distribution.

    Parameters
    ----------
    z_dim : int
        Dimensionality of the latent variable at each time step.
    rnn_dim : int
        Dimensionality of the hidden state in the RNN.
    """

    def __init__(self, z_dim: int, rnn_dim: int) -> None:
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim * 2)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim * 2, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim * 2, z_dim)

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(
        self, z_t_1: torch.Tensor, h_rnn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loc, scale)`` of q(z_t | z_{t-1}, x_{t:T})."""
        h_latent = self.tanh(self.lin_z_to_hidden(z_t_1))
        h_combined = 0.5 * (h_latent + h_rnn)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale


class EmitterRegressor(nn.Module):
    """
    Parameterizes the Gaussian observation likelihood p(y_t | z_t, x_t).

    Generates ``(mu, sigma)`` for the observation at time ``t``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of ``x_t``.
    z_dim : int
        Dimensionality of the latent state ``z_t``.
    emission_dim : int
        Hidden-layer size used to process ``z_t`` and ``x_t``.
    output_dim : int
        Dimensionality of the output ``y_t``.
    """

    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        emission_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.lin_z_to_hidden_mu = nn.Linear(z_dim, emission_dim)
        self.lin_x_to_hidden_mu = nn.Linear(input_dim, emission_dim)
        self.lin_hidden_to_hidden_mu = nn.Linear(emission_dim * 2, emission_dim)
        self.lin_hidden_to_output_mu = nn.Linear(emission_dim, output_dim)

        self.lin_z_to_hidden_sigma = nn.Linear(z_dim, emission_dim)
        self.lin_x_to_hidden_sigma = nn.Linear(input_dim, emission_dim)
        self.lin_hidden_to_hidden_sigma = nn.Linear(emission_dim * 2, emission_dim)
        self.lin_hidden_to_output_sigma = nn.Linear(emission_dim, output_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(
        self, z_t: torch.Tensor, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, sigma)`` of p(y_t | z_t, x_t)."""
        mu = self._predict_mean(z_t, x_t)
        sigma = self._predict_sigma(z_t, x_t)
        return mu, sigma

    def _predict_mean(self, z_t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        h_z = self.relu(self.lin_z_to_hidden_mu(z_t))
        h_x = self.relu(self.lin_x_to_hidden_mu(x_t))
        h1 = self.relu(torch.cat((h_z, h_x), dim=1))
        h2 = self.relu(self.lin_hidden_to_hidden_mu(h1))
        return self.lin_hidden_to_output_mu(h2)

    def _predict_sigma(self, z_t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        h_z = self.relu(self.lin_z_to_hidden_sigma(z_t))
        h_x = self.relu(self.lin_x_to_hidden_sigma(x_t))
        h1 = self.relu(torch.cat((h_z, h_x), dim=1))
        h2 = self.relu(self.lin_hidden_to_hidden_sigma(h1))
        return self.softplus(self.lin_hidden_to_output_sigma(h2))


class Transition(nn.Module):
    """
    Gaussian latent-state transition p(z_t | z_{t-1}, x_t).

    Implements a gated linear update with proposed mean / direct identity
    pathways and a softplus-positive scale.

    Parameters
    ----------
    z_dim : int
        Size of the latent state ``z_t``.
    input_dim : int
        Size of the input ``x_t``.
    transition_dim : int
        Size of the hidden layer for the transformations.
    """

    def __init__(self, z_dim: int, input_dim: int, transition_dim: int) -> None:
        super().__init__()
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_x_to_hidden = nn.Linear(input_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)

        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_x_to_hidden = nn.Linear(input_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)

        self.lin_sig = nn.Linear(z_dim + input_dim, z_dim)

        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_x_to_loc = nn.Linear(input_dim, z_dim)

        # Identity-init the z→loc projection so the layer starts as a passthrough.
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(
        self, z_t_1: torch.Tensor, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loc, scale)`` of p(z_t | z_{t-1}, x_t)."""
        gate = self._compute_gate(z_t_1, x_t)
        proposed_mean = self._compute_proposed_mean(z_t_1, x_t)
        loc = self._compute_loc(z_t_1, x_t, gate)
        scale = self._compute_scale(proposed_mean, x_t)
        return loc, scale

    def _compute_gate(self, z_t_1: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        gate_z = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate_x = self.relu(self.lin_gate_x_to_hidden(x_t))
        return torch.sigmoid(self.lin_gate_hidden_to_z(gate_z + gate_x))

    def _compute_proposed_mean(
        self, z_t_1: torch.Tensor, x_t: torch.Tensor
    ) -> torch.Tensor:
        mean_z = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        mean_x = self.relu(self.lin_proposed_mean_x_to_hidden(x_t))
        return self.lin_proposed_mean_hidden_to_z(mean_z + mean_x)

    def _compute_loc(
        self, z_t_1: torch.Tensor, x_t: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        loc_z = self.lin_z_to_loc(z_t_1)
        loc_x = self.lin_x_to_loc(x_t)
        return (1 - gate) * loc_z + gate * (loc_z + loc_x)

    def _compute_scale(
        self, proposed_mean: torch.Tensor, x_t: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat((self.relu(proposed_mean), x_t), dim=1)
        return self.softplus(self.lin_sig(combined))
