"""Transformer encoder-decoder forecaster.

Implements feature 11 of the v0.4.0 restructure: a multi-head self-attention
encoder paired with a cross-attention decoder for multi-step time-series
forecasting. Uses sinusoidal positional encoding and ``batch_first=True``
PyTorch building blocks throughout.

Teacher forcing convention (matches ``LSTMForecaster``): when ``y`` is
provided to ``forward``, the decoder consumes ``[x[:, -1:], y[:, :-1]]`` in
parallel under a causal self-attention mask and predicts ``y[:, :]`` in one
shot. When ``y`` is ``None``, the decoder rolls out autoregressively starting
from ``x[:, -1:]``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from stockpy.base import EncoderDecoderForecaster
from stockpy.preprocessing import unpack_data

__all__ = ["TransformerForecaster"]


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for ``(batch, seq, d_model)`` inputs.

    The fixed ``pe`` table is registered as a buffer so it follows the module
    onto GPU and through ``state_dict`` round-trips without being treated as a
    learned parameter.

    Parameters
    ----------
    d_model : int
        Embedding dimensionality. Must be even.
    max_len : int, default=5000
        Maximum sequence length supported.
    dropout : float, default=0.1
        Dropout applied to the encoded tensor.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer("pe", self._build_table(d_model, max_len))

    @staticmethod
    def _build_table(d_model: int, max_len: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` (shape ``(batch, seq, d_model)``)."""
        return self.dropout(x + self.pe[: x.size(1)])


class TransformerModel(nn.Module):
    """Encoder-decoder Transformer for multi-step forecasting.

    Composes input projection → positional encoding → ``TransformerEncoder``
    → ``TransformerDecoder`` → output projection. All sub-modules use
    ``batch_first=True``.

    Notes
    -----
    ``nhead`` must divide ``d_model`` evenly (PyTorch constraint).
    """

    def __init__(
        self,
        n_features: int,
        pred_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        context_len: int = 20,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.pred_len = pred_len
        self.d_model = d_model

        max_len = max(context_len, pred_len) + 1
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.encoder = self._build_encoder(
            d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation
        )
        self.decoder = self._build_decoder(
            d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation
        )
        self.output_proj = nn.Linear(d_model, n_features)

    @staticmethod
    def _build_encoder(
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=num_layers)

    @staticmethod
    def _build_decoder(
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ) -> nn.TransformerDecoder:
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        return nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode ``x`` then decode ``pred_len`` future steps.

        Parameters
        ----------
        x : torch.Tensor, shape ``(batch, context_len, n_features)``
            Past context.
        y : torch.Tensor or None, shape ``(batch, pred_len, n_features)``
            Ground-truth future values for parallel teacher forcing. When
            ``None`` the decoder runs autoregressively.
        """
        memory = self._encode(x)
        if y is None:
            return self._decode_autoregressive(x, memory)
        return self._decode_teacher_forced(x, memory, y)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pos_enc(self.input_proj(x)))

    def _decode_teacher_forced(
        self, x: torch.Tensor, memory: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        decoder_input = torch.cat([x[:, -1:, :], y[:, :-1, :]], dim=1)
        emb = self.pos_enc(self.input_proj(decoder_input))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_len).to(
            x.device
        )
        out = self.decoder(emb, memory, tgt_mask=causal_mask)
        return self.output_proj(out)

    def _decode_autoregressive(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        next_step = x[:, -1:, :]
        history = [next_step]
        for _ in range(self.pred_len):
            decoder_input = torch.cat(history, dim=1)
            emb = self.pos_enc(self.input_proj(decoder_input))
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                emb.size(1)
            ).to(x.device)
            out = self.decoder(emb, memory, tgt_mask=causal_mask)
            next_step = self.output_proj(out[:, -1:, :])
            history.append(next_step)
        return torch.cat(history[1:], dim=1)


class TransformerForecaster(EncoderDecoderForecaster):
    """Transformer encoder-decoder forecaster.

    Parameters
    ----------
    d_model : int, default=64
        Dimensionality of the transformer model. Must be divisible by ``nhead``.
    nhead : int, default=4
        Number of attention heads.
    num_encoder_layers : int, default=2
        Number of stacked encoder layers.
    num_decoder_layers : int, default=2
        Number of stacked decoder layers.
    dim_feedforward : int, default=256
        Feed-forward sub-layer width.
    dropout : float, default=0.1
        Dropout rate.
    activation : str, default='relu'
        Activation function in the feed-forward sub-layer.
    context_len : int, default=20
        Encoder window length.
    pred_len : int, default=1
        Decoder window length.
    **kwargs
        Forwarded to :class:`EncoderDecoderForecaster`.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        context_len: int = 20,
        pred_len: int = 1,
        **kwargs,
    ) -> None:
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        super().__init__(context_len=context_len, pred_len=pred_len, **kwargs)
        self._modules = ["module"]

    def __sklearn_tags__(self):
        from sklearn.utils._tags import InputTags, Tags, TargetTags

        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True, multi_output=True),
            input_tags=InputTags(two_d_array=True),
        )

    def _get_tags(self) -> dict:
        return {"requires_y": True}

    def initialize_module(self) -> "TransformerForecaster":
        """Build :class:`TransformerModel` from ``self.n_features_in_``."""
        self.module_ = TransformerModel(
            n_features=self.n_features_in_,
            pred_len=self.pred_len,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            context_len=self.context_len,
        )
        self.criterion_ = nn.MSELoss()
        return self

    def _set_training(self, training: bool = True) -> None:
        self.module_.train(training)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Delegate to the underlying :class:`TransformerModel`."""
        return self.module_(x, y=y)

    def train_step_single(self, batch, **fit_params) -> dict:
        """Pass targets through to enable parallel teacher forcing."""
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        if not self.prob:
            y_pred = self.infer(Xi, y=yi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)
            loss.backward()
            return {"loss": loss, "y_pred": y_pred}
        return super().train_step_single(batch, **fit_params)

    def state_dict(self) -> dict:
        return self.module_.state_dict()

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        self.module_.load_state_dict(state_dict, strict=strict)

    @property
    def model_type(self) -> str:
        # Returns "rnn" so base.py's get_dataset() routes through
        # TimeSeriesDataset (the registry only has "rnn"/"cnn" keys today;
        # both map to TimeSeriesDataset). The string is a data-iterator
        # selector, not an architecture label — adding a "transformer" key
        # to base.py is out of scope for this issue.
        return "rnn"
