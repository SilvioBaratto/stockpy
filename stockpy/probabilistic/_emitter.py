import torch
import torch.nn as nn


class EmitterRegressor(nn.Module):
    """
    The `EmitterRegressor` module parameterizes the Gaussian observation likelihood
    `p(y_t | z_t, x_t)`, essentially modeling the conditional probability of the
    observed data `y_t` given the latent state `z_t` and possibly an additional input `x_t`.

    This module generates parameters for a Gaussian distribution, predicting the mean
    and standard deviation for the observation at time `t`.

    Attributes
    ----------
    lin_z_to_hidden_mu : nn.Linear
        Linear transformation from the latent state to the hidden layer for the mean.
    lin_x_to_hidden_mu : nn.Linear
        Linear transformation from the input to the hidden layer for the mean.
    lin_hidden_to_hidden_mu : nn.Linear
        Linear transformation for processing concatenated hidden states for the mean.
    lin_hidden_to_output_mu : nn.Linear
        Linear transformation from the hidden layer to the output for the mean.
    lin_z_to_hidden_sigma : nn.Linear
        Linear transformation from the latent state to the hidden layer for the standard deviation.
    lin_x_to_hidden_sigma : nn.Linear
        Linear transformation from the input to the hidden layer for the standard deviation.
    lin_hidden_to_hidden_sigma : nn.Linear
        Linear transformation for processing concatenated hidden states for the standard deviation.
    lin_hidden_to_output_sigma : nn.Linear
        Linear transformation from the hidden layer to the output for the standard deviation.
    relu : nn.ReLU
        Rectified Linear Unit activation function.
    softplus : nn.Softplus
        Softplus activation function to ensure the standard deviation is positive.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input `x_t`.
    z_dim : int
        The dimensionality of the latent state `z_t`.
    emission_dim : int
        The size of the hidden layer used to process `z_t` and `x_t`.
    output_dim : int
        The dimensionality of the output `y_t`.
    """

    def __init__(self, input_dim: int, z_dim: int, emission_dim: int, output_dim: int):
        super().__init__()
        # Initialize linear layers for predicting the mean (mu).
        self.lin_z_to_hidden_mu = nn.Linear(z_dim, emission_dim)
        self.lin_x_to_hidden_mu = nn.Linear(input_dim, emission_dim)
        self.lin_hidden_to_hidden_mu = nn.Linear(
            emission_dim * 2, emission_dim
        )  # Concatenated z and x.
        self.lin_hidden_to_output_mu = nn.Linear(emission_dim, output_dim)

        # Initialize linear layers for predicting the standard deviation (sigma).
        self.lin_z_to_hidden_sigma = nn.Linear(z_dim, emission_dim)
        self.lin_x_to_hidden_sigma = nn.Linear(input_dim, emission_dim)
        self.lin_hidden_to_hidden_sigma = nn.Linear(
            emission_dim * 2, emission_dim
        )  # Concatenated z and x.
        self.lin_hidden_to_output_sigma = nn.Linear(emission_dim, output_dim)

        # Initialize non-linearities.
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t: torch.Tensor, x_t: torch.Tensor):
        """
        Perform the forward pass for the EmitterRegressor.

        Processes `z_t` and `x_t` through distinct pathways, each with a linear layer
        and ReLU activation. The results are concatenated and passed through additional
        layers to predict the mean (`mu`) and softplus-transformed standard deviation
        (`sigma`) for the Gaussian distribution of the observation `y_t`.

        Parameters
        ----------
        z_t : torch.Tensor
            The latent state at time `t`.
        x_t : torch.Tensor
            The additional input at time `t`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple of two tensors representing the predicted mean (`mu`) and standard
            deviation (`sigma`) for the Gaussian distribution of `y_t`.
        """

        # Process the latent and input states through their respective pathways for the mean.
        h1_mu_z = self.relu(self.lin_z_to_hidden_mu(z_t))
        h1_mu_x = self.relu(self.lin_x_to_hidden_mu(x_t))
        h1_mu = self.relu(torch.cat((h1_mu_z, h1_mu_x), dim=1))
        h2_mu = self.relu(self.lin_hidden_to_hidden_mu(h1_mu))
        mu = self.lin_hidden_to_output_mu(h2_mu)

        # Process the latent and input states through their respective pathways for the standard deviation.
        h1_sigma_z = self.relu(self.lin_z_to_hidden_sigma(z_t))
        h1_sigma_x = self.relu(self.lin_x_to_hidden_sigma(x_t))
        h1_sigma = self.relu(torch.cat((h1_sigma_z, h1_sigma_x), dim=1))
        h2_sigma = self.relu(self.lin_hidden_to_hidden_sigma(h1_sigma))
        sigma_pre = self.lin_hidden_to_output_sigma(h2_sigma)
        sigma = self.softplus(sigma_pre)  # Ensure that sigma is positive

        return mu, sigma
