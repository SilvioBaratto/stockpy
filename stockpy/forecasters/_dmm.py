from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.nn import PyroModule

from stockpy.base import EncoderDecoderForecaster
from stockpy.preprocessing import unpack_data
from stockpy.forecasters._dmm_components import Combiner, EmitterRegressor, Transition

if TYPE_CHECKING:
    from sklearn.utils._tags import Tags

__all__ = ["DMMForecaster"]


class DMMModel(PyroModule):
    """Deep Markov Model for encoder-decoder time-series forecasting.

    The inference network (RNN + Combiner) acts as the encoder,
    reading past observations to infer latent states. The generative
    network (Transition + Emitter) acts as the decoder, producing
    future predictions autoregressively.

    Parameters
    ----------
    n_features : int
        Number of input/output features.
    pred_len : int
        Number of future time steps to predict.
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
        Dropout rate (unused, kept for API compatibility).
    variance : float, default=0.1
        Variance for distributions (unused, kept for API compatibility).
    activation : str, default='relu'
        Activation function.
    bias : bool, default=True
        Use bias terms.
    context_len : int, default=20
        Encoder window length.
    """

    # Sub-modules below are populated by ``initialize_module``; declared at the
    # class level so static analysis sees their concrete types.
    emitter: EmitterRegressor
    transition: Transition
    combiner: Combiner
    rnn: nn.GRU
    z_0: nn.Parameter
    z_q_0: nn.Parameter
    h_0: nn.Parameter

    def __init__(
        self,
        n_features: int,
        pred_len: int,
        z_dim: int = 32,
        emission_dim: int = 32,
        transition_dim: int = 32,
        rnn_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        variance: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
        context_len: int = 20,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.pred_len = pred_len
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.variance = variance
        self.activation = activation
        self.bias = bias
        self.context_len = context_len

    def initialize_module(self, n_features: int) -> None:
        """Create sub-modules and initial parameters."""
        self.emitter = EmitterRegressor(
            n_features, self.z_dim, self.emission_dim, n_features
        )
        self.transition = Transition(self.z_dim, n_features, self.transition_dim)
        self.combiner = Combiner(self.z_dim, self.rnn_dim)
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=self.rnn_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=self.num_layers,
            bias=self.bias,
        )
        self.z_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(1, self.z_dim))
        self.h_0 = nn.Parameter(torch.zeros(self.num_layers * 2, 1, self.rnn_dim))

    def model(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        annealing_factor: float = 1.0,
    ) -> None:
        """Generative model over context + future.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, context_len, n_features)
            Past observations.
        y : torch.Tensor, shape (batch, pred_len, n_features)
            Future observations.
        annealing_factor : float, default=1.0
            KL-divergence annealing factor.
        """
        full_seq = torch.cat([x, y], dim=1)
        T = full_seq.size(1)

        pyro.module("dmm", self)
        z_prev = self.z_0.expand(x.size(0), self.z_dim)

        with pyro.plate("z_minibatch", len(x)):
            for t in pyro.markov(range(1, T + 1)):
                z_loc, z_scale = self.transition(z_prev, full_seq[:, t - 1, :])
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        dist.Normal(z_loc, z_scale).to_event(1),
                    )
                mu, sigma = self.emitter(z_t, full_seq[:, t - 1, :])
                pyro.sample(
                    "obs_y_%d" % t,
                    dist.Normal(mu, sigma).to_event(1),
                    obs=full_seq[:, t - 1, :],
                )
                z_prev = z_t

    def guide(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        annealing_factor: float = 1.0,
    ) -> None:
        """Variational guide (inference network).

        Parameters
        ----------
        x : torch.Tensor, shape (batch, context_len, n_features)
            Past observations.
        y : torch.Tensor or None, shape (batch, pred_len, n_features)
            Future observations. If None, only context is processed.
        annealing_factor : float, default=1.0
            KL-divergence annealing factor.
        """
        full_seq = torch.cat([x, y], dim=1) if y is not None else x
        T = full_seq.size(1)

        pyro.module("dmm", self)
        h_0 = self.h_0.expand(
            self.num_layers * 2, full_seq.size(0), self.rnn_dim
        ).contiguous()
        rnn_output, _ = self.rnn(full_seq, h_0)

        z_prev = self.z_q_0.expand(full_seq.size(0), self.z_dim)

        with pyro.plate("z_minibatch", len(full_seq)):
            for t in pyro.markov(range(1, T + 1)):
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        dist.Normal(z_loc, z_scale).to_event(1),
                    )
                z_prev = z_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Autoregressive prediction from context.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, context_len, n_features)
            Past observations.

        Returns
        -------
        torch.Tensor, shape (batch, pred_len, n_features)
            Forecasted values.
        """
        batch_size = x.size(0)
        h_0 = self.h_0.expand(
            self.num_layers * 2, batch_size, self.rnn_dim
        ).contiguous()
        rnn_output, _ = self.rnn(x, h_0)

        z_prev = self.z_q_0.expand(batch_size, self.z_dim)
        for t in range(1, x.size(1) + 1):
            z_loc, _ = self.combiner(z_prev, rnn_output[:, t - 1, :])
            z_t = z_loc
            z_prev = z_t

        preds = []
        x_t = x[:, -1, :]

        for _ in range(self.pred_len):
            mu, _ = self.emitter(z_t, x_t)
            preds.append(mu)
            z_loc, _ = self.transition(z_t, x_t)
            z_t = z_loc
            x_t = mu

        return torch.stack(preds, dim=1)


class DMMForecaster(EncoderDecoderForecaster):
    """Deep Markov Model encoder-decoder forecaster.

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
        z_dim: int = 32,
        emission_dim: int = 32,
        transition_dim: int = 32,
        rnn_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        variance: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
        context_len: int = 20,
        pred_len: int = 1,
        **kwargs: Any,
    ) -> None:
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
        self.prob = True
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

    def initialize_module(self) -> DMMForecaster:
        """Create the DMM encoder-decoder module."""
        self.module_ = DMMModel(
            n_features=self.n_features_in_,
            pred_len=self.pred_len,
            z_dim=self.z_dim,
            emission_dim=self.emission_dim,
            transition_dim=self.transition_dim,
            rnn_dim=self.rnn_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            variance=self.variance,
            activation=self.activation,
            bias=self.bias,
            context_len=self.context_len,
        )
        self.module_.initialize_module(self.n_features_in_)
        return self

    def initialize_optimizer(
        self, triggered_directly: bool | None = None
    ) -> DMMForecaster:
        """Override to create a Pyro optimizer for SVI."""
        from pyro.optim import Adam  # type: ignore[attr-defined]

        optim_args = {"lr": getattr(self, "lr", 0.01)}
        self.optimizer_ = Adam(optim_args)
        return self

    def model(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        annealing_factor: float = 1.0,
    ) -> None:
        """Generative model for SVI."""
        return self.module_.model(x, y, annealing_factor)

    def guide(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        annealing_factor: float = 1.0,
    ) -> None:
        """Variational guide for SVI."""
        return self.module_.guide(x, y, annealing_factor)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through the DMM model.

        Parameters
        ----------
        x : torch.Tensor
            Input context window.
        y : torch.Tensor or None
            Unused; kept for API compatibility.

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, pred_len, n_features).
        """
        return self.module_.forward(x)

    def _set_training(self, training: bool = True) -> None:
        """Override to set training mode on the underlying Pyro module."""
        self.module_.train(training)

    def train_step_single(self, batch: Any, **fit_params: Any) -> dict[str, Any]:
        """Override to pass targets to ``infer`` for y_pred logging."""
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        if not self.prob:
            y_pred = self.infer(Xi, y=yi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)
            loss.backward()
            return {"loss": loss, "y_pred": y_pred}
        return super().train_step_single(batch, **fit_params)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the state dict of the underlying Pyro module."""
        return self.module_.state_dict()

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        """Load a state dict into the underlying Pyro module."""
        self.module_.load_state_dict(state_dict, strict=strict)

    @property
    def model_type(self) -> str:
        return "rnn"
