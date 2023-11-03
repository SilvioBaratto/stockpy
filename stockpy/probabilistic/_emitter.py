import torch
import torch.nn as nn
import torch.nn.functional as F

from stockpy.utils import get_activation_function

class EmitterRegressor(nn.Module):
    """
    Parameterizes the Gaussian observation likelihood `p(y_t | z_t, x_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim, output_dim):
        super().__init__()
        
        # Linear transformations for mean (mu)
        self.lin_z_to_hidden_mu = nn.Linear(z_dim, emission_dim)
        self.lin_x_to_hidden_mu = nn.Linear(input_dim, emission_dim)
        self.lin_hidden_to_hidden_mu = nn.Linear(emission_dim * 2, emission_dim)  # Concatenated z and x
        self.lin_hidden_to_output_mu = nn.Linear(emission_dim, output_dim)

        # Linear transformations for standard deviation (sigma)
        self.lin_z_to_hidden_sigma = nn.Linear(z_dim, emission_dim)
        self.lin_x_to_hidden_sigma = nn.Linear(input_dim, emission_dim)
        self.lin_hidden_to_hidden_sigma = nn.Linear(emission_dim * 2, emission_dim)  # Concatenated z and x
        self.lin_hidden_to_output_sigma = nn.Linear(emission_dim, output_dim)

        # Non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t, x_t):
        # For mean (mu)
        h1_mu_z = self.relu(self.lin_z_to_hidden_mu(z_t))
        h1_mu_x = self.relu(self.lin_x_to_hidden_mu(x_t))
        h1_mu = self.relu(torch.cat((h1_mu_z, h1_mu_x), dim=1))
        h2_mu = self.relu(self.lin_hidden_to_hidden_mu(h1_mu))
        mu = self.lin_hidden_to_output_mu(h2_mu)

        # For standard deviation (sigma)
        h1_sigma_z = self.relu(self.lin_z_to_hidden_sigma(z_t))
        h1_sigma_x = self.relu(self.lin_x_to_hidden_sigma(x_t))
        h1_sigma = self.relu(torch.cat((h1_sigma_z, h1_sigma_x), dim=1))
        h2_sigma = self.relu(self.lin_hidden_to_hidden_sigma(h1_sigma))
        sigma_pre = self.lin_hidden_to_output_sigma(h2_sigma)
        sigma = self.softplus(sigma_pre)  # Ensure that sigma is positive

        return mu, sigma
    
class EmitterClassifier(nn.Module):
    """
    Parameterizes the categorical observation likelihood `p(y_t | z_t, x_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim, n_classes):
        super().__init__()
        
        # Linear transformations
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_x_to_hidden = nn.Linear(input_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim * 2, emission_dim)  # Concatenated z and x
        self.lin_hidden_to_output = nn.Linear(emission_dim, n_classes)  # output_dim = n_classes for classification

        # Non-linearities
        self.relu = nn.ReLU()
        
    def forward(self, z_t, x_t):
        # Transform z_t and x_t to hidden states
        h1_z = self.relu(self.lin_z_to_hidden(z_t))
        h1_x = self.relu(self.lin_x_to_hidden(x_t))

        # Concatenate and further transform
        h1 = self.relu(torch.cat((h1_z, h1_x), dim=1))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        
        # Output layer with softmax activation
        output_logits = self.lin_hidden_to_output(h2)
        output_probs = torch.softmax(output_logits, dim=1)  # Softmax to get probabilities
        
        return output_probs