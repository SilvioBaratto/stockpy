import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from stockpy.base import EncoderDecoderForecaster

__all__ = ['DMMForecaster']


class DMMForecaster(EncoderDecoderForecaster):
    """
    Stub for Deep Markov Model (DMM) encoder-decoder forecaster.

    Parameters
    ----------
    z_dim : int, default=32
        Dimensionality of latent states.
    emission_dim : int, default=32
        Dimensionality of emission parameters.
    transition_dim : int, default=32
        Dimensionality of transition parameters.
    rnn_dim : int, default=32
        Dimensionality of RNN hidden states.
    num_layers : int, default=1
        Number of RNN layers.
    dropout : float, default=0.2
        Dropout rate.
    variance : float, default=0.1
        Variance for distributions.
    activation : str, default='relu'
        Activation function.
    bias : bool, default=True
        Use bias terms.
    context_len : int, default=20
        Encoder window length.
    pred_len : int, default=1
        Decoder window length.
    **kwargs
        Passed to EncoderDecoderForecaster.
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
        activation='relu',
        bias=True,
        context_len=20,
        pred_len=1,
        **kwargs,
    ):
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.variance = variance
        self.activation = activation
        self.bias = bias

        super().__init__(
            context_len=context_len,
            pred_len=pred_len,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder forward pass."""
        raise NotImplementedError(
            "DMMForecaster.forward is not yet implemented."
        )

    @property
    def model_type(self):
        return 'rnn'
