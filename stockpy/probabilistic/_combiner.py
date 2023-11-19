import torch
import torch.nn as nn
import torch.nn.functional as F

class Combiner(nn.Module):
    """
    Parameterizes the variational distribution q(z_t | z_{t-1}, x_{t:T}).

    This module maps the input and previous latent state to the parameters of the 
    current latent state's distribution, facilitating backpropagation through time 
    in variational autoencoders applied to time-series data.

    Parameters
    ----------
    z_dim : int
        Dimensionality of the latent variable at each time step.
    rnn_dim : int
        Dimensionality of the hidden state in the RNN.

    Attributes
    ----------
    lin_z_to_hidden : torch.nn.Linear
        Linear layer mapping the latent state at `t-1` to the hidden state dimension.
    lin_hidden_to_loc : torch.nn.Linear
        Linear layer mapping the hidden state to the location (mean) parameter of 
        the latent space at `t`.
    lin_hidden_to_scale : torch.nn.Linear
        Linear layer mapping the hidden state to the scale (standard deviation) parameter 
        of the latent space at `t`.
    tanh : torch.nn.Tanh
        The hyperbolic tangent non-linearity.
    softplus : torch.nn.Softplus
        The softplus non-linearity, which ensures the scale parameter is positive.

    Notes
    -----
    - This class is often used in the construction of the guide for variational 
      inference in time-series models, where it is important to capture the 
      temporal dynamics of the latent variables.
    - The attributes represent neural network components that are responsible for 
      the variational distribution's parameters at each time step, which are 
      optimized during training.
    """

    def __init__(self, z_dim: int, rnn_dim: int):
        super().__init__()
        # Initialize the three linear transformations used in the neural network.
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim * 2)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim * 2, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim * 2, z_dim)
        
        # Initialize the two non-linearities used in the neural network.
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Perform the forward pass to compute the parameters of the Gaussian variational distribution 
        for the latent variable at time t.

        Parameters
        ----------
        z_t_1 : torch.Tensor
            The latent variable from the previous time step (t-1).
        h_rnn : torch.Tensor
            The RNN hidden state that encapsulates information from the current and all future observations.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two tensors: 
            - The first tensor is the location (mean) parameter of the distribution for z_t.
            - The second tensor is the scale (standard deviation) parameter of the distribution for z_t.

        Notes
        -----
        - The forward pass uses the previous latent state and the current hidden state to produce 
        the parameters for the current latent state's distribution.
        - It assumes that both `z_t_1` and `h_rnn` are outputs of appropriate dimensionality, 
        consistent with `z_dim` and `rnn_dim` specified during initialization.
        - The computed scale parameters are constrained to be positive using the softplus function.
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