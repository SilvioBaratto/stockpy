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
        self.lin_hidden_to_hidden_mu = nn.Linear(emission_dim * 2, emission_dim)  # Concatenated z and x.
        self.lin_hidden_to_output_mu = nn.Linear(emission_dim, output_dim)

        # Initialize linear layers for predicting the standard deviation (sigma).
        self.lin_z_to_hidden_sigma = nn.Linear(z_dim, emission_dim)
        self.lin_x_to_hidden_sigma = nn.Linear(input_dim, emission_dim)
        self.lin_hidden_to_hidden_sigma = nn.Linear(emission_dim * 2, emission_dim)  # Concatenated z and x.
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
    
class EmitterClassifier(nn.Module):
    """
    Parameterizes the categorical observation likelihood `p(y_t | z_t, x_t)` for classification.

    Predicts class probabilities for observed data `y_t` at each time step `t`, given 
    the latent state `z_t` and an additional input `x_t`. The logits for each class are 
    computed and converted to probabilities using a softmax function.

    Attributes
    ----------
    lin_z_to_hidden : nn.Linear
        Linear layer mapping the latent state to the hidden layer.
    lin_x_to_hidden : nn.Linear
        Linear layer mapping the input to the hidden layer.
    lin_hidden_to_hidden : nn.Linear
        Linear layer for processing concatenated hidden states.
    lin_hidden_to_output : nn.Linear
        Linear layer mapping the hidden layer to the output logits.
    relu : nn.ReLU
        Activation function applied after linear transformations.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input `x_t`.
    z_dim : int
        Dimensionality of the latent state `z_t`.
    emission_dim : int
        Size of the hidden layer for `z_t` and `x_t` processing.
    n_classes : int
        Number of classes for classification.
    """

    def __init__(self, input_dim: int, z_dim: int, emission_dim: int, n_classes: int):
        super().__init__()
        # Initialize linear layers for processing the latent state and input.
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_x_to_hidden = nn.Linear(input_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim * 2, emission_dim)  # Process concatenated z and x.
        self.lin_hidden_to_output = nn.Linear(emission_dim, n_classes)  # Output layer for class logits.

        # Initialize non-linearity.
        self.relu = nn.ReLU()

    def forward(self, z_t: torch.Tensor, x_t: torch.Tensor):
        """
        Compute forward pass for the EmitterClassifier.

        Transforms the latent state `z_t` and input `x_t` through distinct linear layers followed by ReLU activations. 
        Concatenates the resulting hidden states and processes through another linear layer. The final hidden state is 
        transformed to logits and converted to class probabilities via softmax.

        Parameters
        ----------
        z_t : torch.Tensor
            Latent state at time step `t`.
        x_t : torch.Tensor
            Additional input at time step `t`.

        Returns
        -------
        torch.Tensor
            Class probabilities for each class at time step `t`.
        """

        # Process the latent and input states through respective linear layers followed by ReLU activation.
        h1_z = self.relu(self.lin_z_to_hidden(z_t))
        h1_x = self.relu(self.lin_x_to_hidden(x_t))

        # Concatenate the processed states and further transform through another linear layer.
        h1 = self.relu(torch.cat((h1_z, h1_x), dim=1))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))

        # Compute output logits and apply softmax to get class probabilities.
        output_logits = self.lin_hidden_to_output(h2)
        output_probs = torch.softmax(output_logits, dim=1)  # Softmax applied along the class dimension.

        return output_probs