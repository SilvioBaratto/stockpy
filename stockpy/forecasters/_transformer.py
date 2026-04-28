import torch

from stockpy.base import EncoderDecoderForecaster

__all__ = ["TransformerForecaster"]


class TransformerForecaster(EncoderDecoderForecaster):
    """
    Stub for Transformer-based encoder-decoder forecaster.

    Parameters
    ----------
    d_model : int, default=64
        Dimensionality of the transformer model.
    nhead : int, default=4
        Number of attention heads.
    num_encoder_layers : int, default=2
        Number of encoder layers.
    num_decoder_layers : int, default=2
        Number of decoder layers.
    dim_feedforward : int, default=256
        Dimension of the feedforward network.
    dropout : float, default=0.1
        Dropout rate.
    activation : str, default='relu'
        Activation function.
    context_len : int, default=20
        Encoder window length.
    pred_len : int, default=1
        Decoder window length.
    **kwargs
        Passed to EncoderDecoderForecaster.
    """

    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        activation="relu",
        context_len=20,
        pred_len=1,
        **kwargs,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        super().__init__(
            context_len=context_len,
            pred_len=pred_len,
            **kwargs,
        )
        self._modules = ["module"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder forward pass."""
        raise NotImplementedError(
            "TransformerForecaster.forward is not yet implemented."
        )

    @property
    def model_type(self):
        return "rnn"
