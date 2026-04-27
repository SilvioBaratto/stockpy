import torch
import torch.nn as nn

from stockpy.base import EncoderDecoderForecaster

__all__ = ['TCNForecaster']


class TCNForecaster(EncoderDecoderForecaster):
    """
    Stub for Temporal Convolutional Network (TCN) encoder-decoder forecaster.

    Parameters
    ----------
    hidden_size : int, default=32
        Size of fully connected layers.
    num_filters : int, default=32
        Number of filters per TCN layer.
    kernel_size : int, default=3
        Kernel size for causal convolutions.
    num_layers : int, default=1
        Number of TCN residual blocks.
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
        hidden_size=32,
        num_filters=32,
        kernel_size=3,
        num_layers=1,
        dropout=0.2,
        activation='relu',
        bias=True,
        context_len=20,
        pred_len=1,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
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
            "TCNForecaster.forward is not yet implemented."
        )

    @property
    def model_type(self):
        return 'cnn'
