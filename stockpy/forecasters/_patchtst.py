"""PatchTST encoder-only forecaster with channel-independent patching.

Reference: Nie et al., ICLR 2023.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from stockpy.base import EncoderDecoderForecaster

if TYPE_CHECKING:
    from sklearn.utils._tags import Tags

__all__ = ["PatchTSTForecaster"]


class _RevIN(nn.Module):
    """Reversible instance norm with optional affine transform.

    Parameters
    ----------
    num_features : int
        Number of channels (features).
    affine : bool, default=True
        Whether to learn per-channel affine parameters.
    eps : float, default=1e-5
        Epsilon for numerical stability.
    """

    def __init__(
        self, num_features: int, affine: bool = True, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def _normalize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize ``x`` over the time axis.

        Parameters
        ----------
        x : torch.Tensor, shape ``(batch, seq_len, num_features)``

        Returns
        -------
        x_norm : torch.Tensor
        mean : torch.Tensor
        std : torch.Tensor
        """
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        std = x.std(dim=1, keepdim=True) + self.eps  # (B, 1, C)
        x_norm = (x - mean) / std
        if self.gamma is not None and self.beta is not None:
            x_norm = x_norm * self.gamma.view(1, 1, -1)
            x_norm = x_norm + self.beta.view(1, 1, -1)
        return x_norm, mean, std

    def _denormalize(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Undo normalisation (and optional affine) on ``x``.

        Parameters
        ----------
        x : torch.Tensor, shape ``(batch, *, num_features)``
        mean : torch.Tensor, shape ``(batch, 1, num_features)``
        std : torch.Tensor, shape ``(batch, 1, num_features)``
        """
        if self.gamma is not None and self.beta is not None:
            x = (x - self.beta.view(1, 1, -1)) / self.gamma.view(1, 1, -1)
        return x * std + mean


class _PatchEmbedding(nn.Module):
    """Unfold + linear projection + learnable positional embedding.

    Parameters
    ----------
    patch_len : int
        Length of each patch.
    d_model : int
        Model dimensionality.
    num_patches : int
        Number of patches (pre-computed after padding).
    """

    def __init__(self, patch_len: int, d_model: int, num_patches: int) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project patches and add positional encoding.

        Parameters
        ----------
        x : torch.Tensor, shape ``(batch, num_patches, patch_len)``

        Returns
        -------
        torch.Tensor, shape ``(batch, num_patches, d_model)``
        """
        x = self.value_embedding(x)  # (B, N, d_model)
        return x + self.pos_embedding


class _PatchTSTModel(nn.Module):
    """Top-level module: RevIN → patch → TransformerEncoder → head.

    Parameters
    ----------
    n_features : int
        Number of input/output features (channels).
    context_len : int
        Input sequence length.
    pred_len : int
        Number of steps to forecast.
    patch_len : int
        Patch size.
    stride : int
        Stride between consecutive patches.
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
    revin : bool
        Whether to apply reversible instance norm.
    revin_affine : bool
        Whether RevIN uses affine parameters.
    """

    def __init__(
        self,
        n_features: int,
        context_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        revin: bool,
        revin_affine: bool,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.context_len = context_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_patches = self._compute_num_patches(context_len, patch_len, stride)

        if revin:
            self.revin = _RevIN(n_features, affine=revin_affine)
        else:
            self.revin = None

        self.patch_embedding = _PatchEmbedding(
            patch_len=patch_len,
            d_model=d_model,
            num_patches=self.num_patches,
        )

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

        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    @staticmethod
    def _compute_num_patches(context_len: int, patch_len: int, stride: int) -> int:
        """Compute number of patches after end-padding."""
        remainder = (context_len - patch_len) % stride
        if remainder == 0:
            return (context_len - patch_len) // stride + 1
        pad_amount = stride - remainder
        padded_len = context_len + pad_amount
        return (padded_len - patch_len) // stride + 1

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
        batch_size = x.size(0)

        # 1. RevIN
        if self.revin is not None:
            x_norm, mean, std = self.revin._normalize(x)
        else:
            x_norm = x
            mean = std = None

        # 2. Permute to (B, C, L)
        x_norm = x_norm.permute(0, 2, 1)  # (B, C, L)

        # 3. End-padding so (L_padded - patch_len) divisible by stride
        remainder = (self.context_len - self.patch_len) % self.stride
        if remainder != 0:
            pad_amount = self.stride - remainder
            # Replicate-pad right edge with last value
            last = x_norm[..., -1:]
            pad = last.repeat(1, 1, pad_amount)
            x_norm = torch.cat([x_norm, pad], dim=-1)

        # 4. Unfold: (B, C, N, patch_len)
        x_norm = x_norm.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x_norm: (B, C, N, patch_len)

        # 5. Channel independence: (B*C, N, patch_len)
        x_norm = x_norm.reshape(-1, x_norm.size(2), self.patch_len)

        # 6. Patch embedding
        emb = self.patch_embedding(x_norm)  # (B*C, N, d_model)

        # 7. Transformer encoder
        enc = self.encoder(emb)  # (B*C, N, d_model)

        # 8. Flatten head
        flat = enc.reshape(batch_size * self.n_features, -1)
        out = self.head(flat)  # (B*C, pred_len)

        # 9. Reshape to (B, C, pred_len) then permute to (B, pred_len, C)
        out = out.view(batch_size, self.n_features, self.pred_len)
        out = out.permute(0, 2, 1)  # (B, pred_len, C)

        # 10. Denormalise
        if self.revin is not None and mean is not None and std is not None:
            out = self.revin._denormalize(out, mean, std)

        return out


class PatchTSTForecaster(EncoderDecoderForecaster):
    """PatchTST encoder-only forecaster.

    Parameters
    ----------
    patch_len : int, default=16
        Length of each patch.
    stride : int, default=8
        Stride between consecutive patches.
    d_model : int, default=128
        Transformer dimensionality. Must be divisible by ``nhead``.
    nhead : int, default=16
        Number of attention heads.
    num_layers : int, default=3
        Number of encoder layers.
    dim_feedforward : int, default=512
        FFN hidden dimension.
    dropout : float, default=0.2
        Dropout rate.
    activation : str, default='gelu'
        Activation function.
    revin : bool, default=True
        Whether to apply reversible instance norm.
    revin_affine : bool, default=True
        Whether RevIN uses affine parameters.
    context_len : int, default=96
        Encoder window length.
    pred_len : int, default=24
        Decoder window length.
    **kwargs
        Forwarded to :class:`EncoderDecoderForecaster`.
    """

    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 16,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        activation: str = "gelu",
        revin: bool = True,
        revin_affine: bool = True,
        context_len: int = 96,
        pred_len: int = 24,
        **kwargs: Any,
    ) -> None:
        if context_len < patch_len:
            raise ValueError(
                f"context_len ({context_len}) must be >= " f"patch_len ({patch_len})"
            )
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        if patch_len < 1:
            raise ValueError(f"patch_len ({patch_len}) must be >= 1")
        if stride < 1:
            raise ValueError(f"stride ({stride}) must be >= 1")

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.revin = revin
        self.revin_affine = revin_affine

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

    def initialize_module(self) -> PatchTSTForecaster:
        """Build :class:`_PatchTSTModel` from ``self.n_features_in_``."""
        self.module_ = _PatchTSTModel(
            n_features=self.n_features_in_,
            context_len=self.context_len,
            pred_len=self.pred_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            revin=self.revin,
            revin_affine=self.revin_affine,
        )
        self.criterion_ = nn.MSELoss()
        return self

    def _set_training(self, training: bool = True) -> None:
        self.module_.train(training)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Delegate to the underlying :class:`_PatchTSTModel`."""
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
