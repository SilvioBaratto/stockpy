import torch
import torch.nn as nn

from stockpy.base import EncoderDecoderForecaster
from stockpy.utils import get_activation_function

__all__ = ['GRUForecaster']


class GRUForecaster(EncoderDecoderForecaster):
    """
    Stub for GRU-based encoder-decoder forecaster.

    Parameters
    ----------
    rnn_size : int, default=32
        Number of features in the hidden state.
    hidden_size : int, default=32
        Size of fully connected layers after the GRU.
    num_layers : int, default=1
        Number of stacked GRU layers.
    dropout : float, default=0.2
        Dropout rate.
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
        rnn_size=32,
        hidden_size=32,
        num_layers=1,
        dropout=0.2,
        activation='relu',
        bias=True,
        context_len=20,
        pred_len=1,
        **kwargs,
    ):
        self.rnn_size = rnn_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
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
            "GRUForecaster.forward is not yet implemented."
        )

    @property
    def model_type(self):
        return 'rnn'
