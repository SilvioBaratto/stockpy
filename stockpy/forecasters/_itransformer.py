"""iTransformer encoder-only forecaster with variates-as-tokens.

Reference: Liu et al., ICLR 2024.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from stockpy.base import EncoderDecoderForecaster

if TYPE_CHECKING:
    from sklearn.utils._tags import Tags

__all__ = ["iTransformerForecaster"]


class _iTransformerEmbedding(nn.Module):
    """Per-variate linear projection L -> d_model.

    Shared weights across all C variates.

    Parameters
    ----------
    context_len : int
        Input sequence length (L).
    d_model : int
        Model dimensionality.
    """

    def __init__(self, context_len: int, d_model: int) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(context_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project each variate independently.

        Parameters
        ----------
        x : torch.Tensor, shape ``(batch, n_features, context_len)``

        Returns
        -------
        torch.Tensor, shape ``(batch, n_features, d_model)``
        """
        return self.value_embedding(x)


class _iTransformerModel(nn.Module):
    """Top-level module. Composes per-variate embedding +
    nn.TransformerEncoder (no posenc) + per-variate projection head.

    Parameters
    ----------
    context_len : int
        Input sequence length.
    pred_len : int
        Number of steps to forecast.
    d_model : int
        Transformer dimensionality.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of encoder layers.
    dim_feedforward : int
        FFN hidden dimension.
    dropout : float
        Dropout rate.
    activation : str
        FFN activation.
    use_norm : bool
        Whether to apply series-wise normalisation.
    """

    def __init__(
        self,
        context_len: int,
        pred_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        use_norm: bool,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.use_norm = use_norm

        self.embedding = _iTransformerEmbedding(context_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Encode ``x`` and forecast ``pred_len`` steps.

        Parameters
        ----------
        x : torch.Tensor, shape ``(batch, context_len, n_features)``
        y : ignored

        Returns
        -------
        torch.Tensor, shape ``(batch, pred_len, n_features)``
        """
        # 1. Series-wise normalisation
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True)  # (B, 1, C)
            std = x.std(dim=1, keepdim=True) + 1e-5  # (B, 1, C)
            x_norm = (x - mean) / std
        else:
            x_norm = x
            mean = std = None

        # 2. Permute to (B, C, L)
        x_norm = x_norm.permute(0, 2, 1)  # (B, C, L)

        # 3. Variate embedding: (B, C, L) -> (B, C, d_model)
        emb = self.embedding(x_norm)  # (B, C, d_model)

        # 4. Transformer encoder (no positional encoding)
        enc = self.encoder(emb)  # (B, C, d_model)

        # 5. Projection head: (B, C, d_model) -> (B, C, pred_len)
        out = self.head(enc)  # (B, C, pred_len)

        # 6. Permute to (B, pred_len, C)
        out = out.permute(0, 2, 1)  # (B, pred_len, C)

        # Denormalise
        if self.use_norm and mean is not None and std is not None:
            out = out * std + mean

        return out


class iTransformerForecaster(EncoderDecoderForecaster):
    """iTransformer encoder-only forecaster.

    Parameters
    ----------
    d_model : int, default=128
        Transformer dimensionality. Must be divisible by ``nhead``.
    nhead : int, default=8
        Number of attention heads.
    num_layers : int, default=3
        Number of encoder layers.
    dim_feedforward : int, default=512
        FFN hidden dimension.
    dropout : float, default=0.1
        Dropout rate.
    activation : str, default='gelu'
        Activation function.
    use_norm : bool, default=True
        Whether to apply series-wise normalisation.
    context_len : int, default=96
        Encoder window length.
    pred_len : int, default=24
        Decoder window length.
    **kwargs
        Forwarded to :class:`EncoderDecoderForecaster`.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = True,
        context_len: int = 96,
        pred_len: int = 24,
        **kwargs: Any,
    ) -> None:
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        if context_len < 1:
            raise ValueError(f"context_len ({context_len}) must be >= 1")
        if pred_len < 1:
            raise ValueError(f"pred_len ({pred_len}) must be >= 1")

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.use_norm = use_norm

        super().__init__(context_len=context_len, pred_len=pred_len, **kwargs)
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

    def initialize_module(self) -> iTransformerForecaster:
        """Build :class:`_iTransformerModel` from stored hyperparameters."""
        self.module_ = _iTransformerModel(
            context_len=self.context_len,
            pred_len=self.pred_len,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            use_norm=self.use_norm,
        )
        self.criterion_ = nn.MSELoss()
        return self

    def _set_training(self, training: bool = True) -> None:
        self.module_.train(training)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Delegate to the underlying :class:`_iTransformerModel`."""
        return self.module_(x, y=y)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module_.state_dict()

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        self.module_.load_state_dict(state_dict, strict=strict)

    @property
    def model_type(self) -> str:
        # "rnn" is the data-iterator selector in base.py; applies to all
        # sequence forecasters regardless of internal architecture.
        return "rnn"
