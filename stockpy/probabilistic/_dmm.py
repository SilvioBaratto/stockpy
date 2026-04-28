import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.nn import PyroModule

from stockpy.base import EncoderDecoderForecaster

from ._combiner import Combiner
from ._emitter import EmitterRegressor
from ._transition import Transition

__all__ = ["DMMRegressor"]


class DMM(PyroModule):
    """
    Deep Markov Model with Markov latent state space.

    Parameters
    ----------
    z_dim : int, optional
        Dimensionality of latent states, by default 32.
    emission_dim : int, optional
        Dimensionality of emission parameters, by default 32.
    transition_dim : int, optional
        Dimensionality of transition parameters, by default 32.
    rnn_dim : int, optional
        Dimensionality of RNN hidden states, by default 32.
    num_layers : int, optional
        Number of RNN layers, by default 1.
    dropout : float, optional
        Dropout rate, by default 0.2.
    variance : float, optional
        Variance for distributions, by default 0.1.
    activation : str, optional
        Neural network activation function, by default 'relu'.
    bias : bool, optional
        Use bias in layers, by default True.
    seq_len : int, optional
        Input sequence length, by default 20.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    z_dim : int
        Latent state dimensionality.
    emission_dim : int
        Emission parameters dimensionality.
    transition_dim : int
        Transition parameters dimensionality.
    rnn_dim : int
        RNN hidden state dimensionality.
    num_layers : int
        RNN layer count.
    dropout : float
        Regularization dropout rate.
    variance : float
        Variance for probability distributions.
    activation : str
        Activation function in network layers.
    bias : bool
        Bias term inclusion in layers.
    seq_len : int
        Sequence length for input data.
    emitter_rgr : PyroModule
        Regression module for emissions.
    transition : PyroModule
        Transition state module.
    combiner : PyroModule
        Module combining RNN output and latent states.
    rnn : torch.nn.Module
        Recurrent neural network module.
    z_0 : torch.nn.Parameter
        Initial latent state parameter.
    z_q_0 : torch.nn.Parameter
        Initial latent state variational parameter.
    h_0 : torch.nn.Parameter
        Initial hidden state parameter for RNN.

    Notes
    -----
    Inherits from PyroModule for integration of deep learning with probabilistic modeling.
    Operates as a regressor using the emitter_rgr module.
    """

    def __init__(
        self,
        z_dim=32,
        emission_dim=32,
        transition_dim=32,
        rnn_dim=32,
        num_layers=1,
        dropout=0.2,
        variance=0.1,
        activation="relu",
        bias=True,
        seq_len=20,
        **kwargs,
    ):

        super().__init__()

        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.variance = variance
        self.activation = activation
        self.bias = bias
        self.seq_len = seq_len

    def initialize_module(self):
        """
        Initialize the DMM neural network modules and parameters.

        Raises
        ------
        TypeError
            If instance is not an `EncoderDecoderForecaster`.
        """

        self.output_size = self.n_outputs_

        self.emitter_rgr = EmitterRegressor(
            self.n_features_in_, self.z_dim, self.emission_dim, self.output_size
        )

        self.transition = Transition(
            self.z_dim, self.n_features_in_, self.transition_dim
        )

        self.combiner = Combiner(self.z_dim, self.rnn_dim)

        self.rnn = nn.GRU(
            input_size=self.n_features_in_,
            hidden_size=self.rnn_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=self.num_layers,
            bias=self.bias,
        )

        # define a (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(1, self.z_dim))
        # define a (trainable) parameter for the initial hidden state of the RNN
        self.h_0 = nn.Parameter(torch.zeros(self.num_layers * 2, 1, self.rnn_dim))

    @property
    def model_type(self):
        return "rnn"


class DMMRegressor(EncoderDecoderForecaster, DMM):
    """
    Specialized DMM for regression tasks using deep generative modeling.

    Attributes
    ----------
    Inherits all attributes from the DMM class with adjustments for regression.

    Parameters
    ----------
    z_dim : int, optional
        Dimensionality of the latent state space (default is 32).
    emission_dim : int, optional
        Dimensionality of the emission parameters (default is 32).
    transition_dim : int, optional
        Dimensionality of the transition parameters (default is 32).
    rnn_dim : int, optional
        Dimensionality of the RNN hidden states (default is 32).
    num_layers : int, optional
        Number of layers in the RNN (default is 1).
    dropout : float, optional
        Dropout rate for regularization (default is 0.2).
    variance : float, optional
        Variance parameter for distributions (default is 0.1).
    activation : str, optional
        Activation function in neural network layers (default is 'relu').
    bias : bool, optional
        If `True`, adds bias to RNN layers (default is True).
    seq_len : int, optional
        Input sequence length (default is 20).
    **kwargs
        Additional arbitrary keyword arguments passed to EncoderDecoderForecaster base class.

    Notes
    -----
    Adjusts DMM output dimensions for regression task requirements.
    """

    def __init__(
        self,
        z_dim=32,
        emission_dim=32,
        transition_dim=32,
        rnn_dim=32,
        num_layers=1,
        dropout=0.2,
        variance=0.1,
        activation="relu",
        bias=True,
        seq_len=20,
        **kwargs,
    ):
        """
        Constructor method for the DMMRegressor class.

        This method initializes a new instance of DMMRegressor with the specified
        parameters or their default values. It first initializes the EncoderDecoderForecaster base class
        and then the DMM class with the provided arguments.

        Raises
        ------
        TypeError
            If an argument is not of the expected type.
        ValueError
            If an invalid value is passed to an argument.
        """

        EncoderDecoderForecaster.__init__(self, **kwargs)
        DMM.__init__(
            self,
            z_dim=z_dim,
            emission_dim=emission_dim,
            transition_dim=transition_dim,
            rnn_dim=rnn_dim,
            num_layers=num_layers,
            dropout=dropout,
            variance=variance,
            activation=activation,
            bias=bias,
            seq_len=seq_len,
            **kwargs,
        )

    def model(self, x, y, annealing_factor=1.0):
        """
        Defines the generative model for the deep Markov model (DMM) regressor.

        Parameters
        ----------
        x : torch.Tensor
            Observed input features with shape (batch_size, T_max, input_dim), where T_max is the
            maximum sequence length in the batch.
        y : torch.Tensor
            Target variable with shape (batch_size, T_max, output_dim), where T_max is the maximum
            sequence length in the batch.
        annealing_factor : float, optional
            Factor to anneal the KL-divergence term in the loss during training (default is 1.0).

        """

        # Determine the sequence length from the input features `x`.
        T_max = x.size(1)

        # Register this module with Pyro so that all its parameters can be optimized.
        pyro.module("dmm", self)

        # Initialize the previous latent state `z_prev` with `z_0`, which is a trainable parameter.
        # This is for the first time step where there's no previous latent state available.
        z_prev = self.z_0.expand(x.size(0), self.z_0.size(0))

        # Start a Pyro plate for batching. This handles the mini-batch processing within Pyro,
        # allowing for vectorized operations across the batch dimension.
        with pyro.plate("z_minibatch", len(x)):
            # Loop over each time step in the sequence. Pyro's markov context manager tells Pyro
            # that the current time step only depends on the previous one, which enables
            # optimizations for markov models.
            for t in pyro.markov(range(1, T_max + 1)):

                # Get the parameters for the distribution of the current latent state `z_t`
                # by passing the previous latent state and the current observed data through
                # the transition module.
                z_loc, z_scale = self.transition(z_prev, x[:, t - 1, :])

                # Scale the contribution of the KL-divergence to the loss function by the annealing factor.
                # This can help in stabilizing training by initially weighting the KL-divergence term less.
                with pyro.poutine.scale(scale=annealing_factor):
                    # Sample the latent variable `z_t` for time step `t` from the normal distribution
                    # parameterized by `z_loc` and `z_scale`. `.to_event(1)` indicates that this
                    # distribution is over a vector-valued random variable.
                    z_t = pyro.sample(
                        "z_%d" % t, dist.Normal(z_loc, z_scale).to_event(1)
                    )

                # Pass the sampled latent state `z_t` and the observed data to the emission module
                # to get the parameters `mu` and `sigma` of the observed distribution.
                mu, sigma = self.emitter_rgr(z_t, x[:, t - 1, :])

                # Sample the observation `y` at time `t` from the normal distribution parameterized
                # by the output of the emission module. The `obs=y[:, t - 1, :]` argument indicates
                # that this sample corresponds to the actual observed data.
                pyro.sample("obs_y_%d" % t, dist.Normal(mu, sigma).to_event(1), obs=y)

                # Update `z_prev` to the current `z_t` to be used in the next time step.
                z_prev = z_t

    def guide(self, x, y=None, annealing_factor=1.0):
        """
        Defines the variational guide for the deep Markov model (DMM) regressor.

        Parameters
        ----------
        x : torch.Tensor
            Observed input features with shape (batch_size, T_max, input_dim), where T_max is the
            maximum sequence length in the batch.
        y : torch.Tensor, optional
            The target variable, not utilized in the guide, included for compatibility with the model
            signature (default is None).
        annealing_factor : float, optional
            Factor to anneal the KL-divergence term in the loss during training (default is 1.0).

        """

        # Determine the sequence length from the input features `x`.
        T_max = x.size(1)
        # Register this module with Pyro, which is necessary for optimization.
        pyro.module("dmm", self)

        # Prepare the initial hidden state for the RNN, ensuring it's compatible with the input's batch size.
        h_0_contig = self.h_0.expand(
            self.num_layers * 2, x.size(0), self.rnn.hidden_size
        ).contiguous()

        # Process the sequence `x` through the RNN to obtain the output for each time step.
        rnn_output, _ = self.rnn(x, h_0_contig)

        # Initialize the previous latent state `z_prev` with `z_q_0`, which is a trainable parameter
        # that represents the initial state of the latent variable.
        z_prev = self.z_q_0.expand(x.size(0), self.z_dim)

        # Start a Pyro plate for batching, similar to the model function.
        with pyro.plate("z_minibatch", len(x)):
            # Loop over each time step in the sequence in reverse order.
            for t in pyro.markov(range(1, T_max + 1)):

                # Use the `combiner` to calculate the parameters of the variational distribution for `z_t`
                # based on the previous latent state `z_prev` and the output of the RNN at the corresponding time step.
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                # Define the normal distribution for the current latent variable `z_t` with the computed mean and scale.
                z_dist = dist.Normal(z_loc, z_scale)

                # Sample `z_t` from the variational distribution without scaling the KL-divergence term.
                # This is in contrast with the model where the annealing factor scales the KL-divergence term.
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, z_dist.to_event(1))

                # Update `z_prev` with the sampled `z_t` to be used for the next time step.
                z_prev = z_t

    def forward(self, x):
        """
        Forward pass for the Deep Markov Model (DMM) regressor to generate predictions.

        Given an input sequence `x`, this method produces the sequence of predictions by
        performing the following steps for each time step:
        1. Retrieves the latent variables `z_t` from the guide trace.
        2. Generates the mean predictions from the emitter regressor using `z_t`.

        The predictions for all time steps are then combined, and the output for the last
        time step is returned as the final prediction for the sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, T_max, input_dim), where `T_max` is the maximum
            sequence length in the batch and `input_dim` is the dimension of input features.

        Returns
        -------
        torch.Tensor
            Output tensor containing predicted values for the last time step in the input sequence,
            with shape (batch_size, output_dim), where `output_dim` is the dimension of the output
            features.

        """
        # Initialize a list to hold the predictions at each time step.
        preds = []

        # Record the operations of the guide function on the input `x` to later access the latent variables.
        guide_trace = pyro.poutine.trace(self.guide).get_trace(x)

        # Determine the sequence length from the input features `x`.
        T_max = x.size(1)

        # Loop over each time step in the mini-batch.
        for t in pyro.markov(range(1, T_max + 1)):

            # Retrieve the value of the latent variable `z_t` from the guide trace for the current time step.
            z_t = guide_trace.nodes[f"z_{t}"]["value"]

            # Use the emitter (which is typically a neural network module) to get the mean prediction at time `t`.
            # The underscore '_' is used to discard the second return value (typically the standard deviation).
            mean_t, _ = self.emitter_rgr(z_t, x[:, t - 1, :])

            # Append the mean prediction for the current time step to the list of predictions.
            preds.append(mean_t)

        # Combine the predictions from all time steps into a single tensor.
        preds = torch.stack(preds)

        # Return the predictions for the last time step for each element in the batch.
        return preds[-1, :, :]

    def predict(self, X, predict_nonlinearity="auto"):
        """
        Forecast future values for the given input sequences.

        Parameters
        ----------
        X : array-like
            Input data for forecasting.
        predict_nonlinearity : callable or None, optional
            Nonlinearity to apply to predictions.

        Returns
        -------
        numpy.ndarray
            Forecasted values.
        """
        return super().predict(X, predict_nonlinearity)
