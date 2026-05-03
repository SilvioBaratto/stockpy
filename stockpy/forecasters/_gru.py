from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from stockpy.base import EncoderDecoderForecaster
from stockpy.preprocessing import unpack_data
from stockpy.utils import get_activation_function

if TYPE_CHECKING:
    from sklearn.utils._tags import Tags

__all__ = ["GRUForecaster"]


class GRUModel(nn.Module):
    """GRU-based encoder-decoder for time-series forecasting.

    Parameters
    ----------
    n_features : int
        Number of input/output features.
    pred_len : int
        Number of future time steps to predict.
    rnn_size : int, default=32
        Hidden size of GRU layers.
    hidden_size : int, default=32
        Hidden size of the output projection MLP.
    num_layers : int, default=1
        Number of stacked GRU layers.
    dropout : float, default=0.0
        Dropout rate (applied when num_layers > 1).
    activation : str, default='relu'
        Activation function for the projection MLP.
    bias : bool, default=True
        Use bias in GRU and linear layers.
    """

    def __init__(
        self,
        n_features: int,
        pred_len: int,
        rnn_size: int = 32,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.pred_len = pred_len
        self.rnn_size = rnn_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = nn.GRU(
            input_size=n_features,
            hidden_size=rnn_size,
            num_layers=num_layers,
            batch_first=True,
            bias=bias,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.GRU(
            input_size=n_features,
            hidden_size=rnn_size,
            num_layers=num_layers,
            batch_first=True,
            bias=bias,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        act = get_activation_function(activation)
        self.output_proj = nn.Sequential(
            nn.Linear(rnn_size, hidden_size, bias=bias),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_features, bias=bias),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, context_len, n_features)
            Past context window.
        y : torch.Tensor or None, shape (batch, pred_len, n_features)
            Ground-truth future values for teacher forcing. If None,
            the decoder runs autoregressively.

        Returns
        -------
        torch.Tensor, shape (batch, pred_len, n_features)
            Forecasted values.
        """
        # Encode context window
        _, hidden = self.encoder(x)
        # hidden: (num_layers, batch, rnn_size)

        if y is not None:
            # Teacher forcing: use ground-truth shifted by one timestep
            # First decoder input is the last timestep of encoder input
            first_input = x[:, -1:, :]  # (batch, 1, n_features)
            decoder_input = torch.cat([first_input, y[:, :-1, :]], dim=1)
            # decoder_input: (batch, pred_len, n_features)
            out, _ = self.decoder(decoder_input, hidden)
            # out: (batch, pred_len, rnn_size)
            return self.output_proj(out)
        else:
            # Autoregressive decoding
            decoder_input = x[:, -1:, :]  # (batch, 1, n_features)
            outputs = []
            for _ in range(self.pred_len):
                out, hidden = self.decoder(decoder_input, hidden)
                # out: (batch, 1, rnn_size)
                pred = self.output_proj(out)  # (batch, 1, n_features)
                outputs.append(pred)
                decoder_input = pred  # feed previous prediction as next input
            return torch.cat(outputs, dim=1)  # (batch, pred_len, n_features)


class GRUForecaster(EncoderDecoderForecaster):
    """GRU-based encoder-decoder forecaster.

    Parameters
    ----------
    rnn_size : int, default=32
        Number of features in the GRU hidden state.
    hidden_size : int, default=32
        Size of the fully-connected projection layer.
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
        rnn_size: int = 32,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        activation: str = "relu",
        bias: bool = True,
        context_len: int = 20,
        pred_len: int = 1,
        **kwargs: Any,
    ) -> None:
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
        self._modules = ["module"]

    def __sklearn_tags__(self) -> Tags:
        from sklearn.utils._tags import InputTags, Tags, TargetTags

        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True, multi_output=True),
            input_tags=InputTags(two_d_array=True),
        )

    def _get_tags(self) -> dict[str, bool]:
        return {"requires_y": True}

    def initialize_module(self) -> GRUForecaster:
        """Create the GRU encoder-decoder module and loss criterion."""
        self.module_ = GRUModel(
            n_features=self.n_features_in_,
            pred_len=self.pred_len,
            rnn_size=self.rnn_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            activation=self.activation,
            bias=self.bias,
        )
        self.criterion_ = nn.MSELoss()
        return self

    def _set_training(self, training: bool = True) -> None:
        """Override to set training mode on the PyTorch module."""
        self.module_.train(training)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through the GRU model.

        Parameters
        ----------
        x : torch.Tensor
            Input context window.
        y : torch.Tensor or None
            Target values for teacher forcing.

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, pred_len, n_features).
        """
        return self.module_(x, y=y)

    def train_step_single(self, batch: Any, **fit_params: Any) -> dict[str, Any]:
        """Override to pass targets to ``infer`` for teacher forcing."""
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        if not self.prob:
            y_pred = self.infer(Xi, y=yi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)
            loss.backward()
            return {"loss": loss, "y_pred": y_pred}
        return super().train_step_single(batch, **fit_params)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the state dict of the underlying PyTorch module."""
        return self.module_.state_dict()

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        """Load a state dict into the underlying PyTorch module."""
        self.module_.load_state_dict(state_dict, strict=strict)

    @property
    def model_type(self) -> str:
        return "rnn"
