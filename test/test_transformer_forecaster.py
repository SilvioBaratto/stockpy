"""Tests for the Transformer encoder-decoder forecaster.

Coverage for issues #33 (model + forecaster implementation) and #34 (broader
behavioural suite mirroring ``test_lstm_forecaster.py``). The canonical
``TestTransformerForecaster`` class lives at the bottom of the file; the
upper classes carry the granular structural and unit-level coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from stockpy.base import EncoderDecoderForecaster
from stockpy.forecasters import TransformerForecaster
from stockpy.forecasters._transformer import (
    TransformerModel,
    _PositionalEncoding,
)


def _fit_zero_epochs(model: TransformerForecaster, n_features: int = 3) -> None:
    X = np.random.randn(60, n_features).astype(np.float32)
    y = np.random.randn(60, n_features).astype(np.float32)
    model.fit(X, y, epochs=0, verbose=0, train_split=None)


class TestPositionalEncoding:
    def test_when_called_returns_input_shape(self):
        pe = _PositionalEncoding(d_model=16, max_len=64, dropout=0.0)
        x = torch.randn(2, 8, 16)
        out = pe(x)
        assert out.shape == x.shape

    def test_when_seq_len_exceeds_max_len_raises(self):
        pe = _PositionalEncoding(d_model=16, max_len=4, dropout=0.0)
        x = torch.randn(1, 8, 16)
        with pytest.raises((IndexError, RuntimeError)):
            pe(x)

    def test_when_built_pe_is_registered_as_buffer(self):
        pe = _PositionalEncoding(d_model=16, max_len=32, dropout=0.0)
        buffer_names = {name for name, _ in pe.named_buffers()}
        assert "pe" in buffer_names

    def test_when_dropout_is_zero_output_matches_x_plus_encoding(self):
        pe = _PositionalEncoding(d_model=8, max_len=16, dropout=0.0)
        x = torch.zeros(1, 4, 8)
        out = pe(x)
        # With x=0 and no dropout, output must equal the first 4 PE rows.
        assert torch.allclose(out[0], pe.pe[:4])


class TestTransformerModelComposition:
    def _build(self, n_features: int = 3, pred_len: int = 4) -> TransformerModel:
        return TransformerModel(
            n_features=n_features,
            pred_len=pred_len,
            d_model=16,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=32,
            dropout=0.0,
            activation="relu",
            context_len=8,
        )

    def test_when_built_has_input_projection(self):
        model = self._build()
        assert isinstance(model.input_proj, nn.Linear)
        assert model.input_proj.in_features == 3
        assert model.input_proj.out_features == 16

    def test_when_built_has_positional_encoding(self):
        model = self._build()
        assert isinstance(model.pos_enc, _PositionalEncoding)

    def test_when_built_has_transformer_encoder_with_correct_layer_count(self):
        model = self._build()
        assert isinstance(model.encoder, nn.TransformerEncoder)
        assert model.encoder.num_layers == 2

    def test_when_built_has_transformer_decoder_with_correct_layer_count(self):
        model = self._build()
        assert isinstance(model.decoder, nn.TransformerDecoder)
        assert model.decoder.num_layers == 2

    def test_when_built_has_output_projection(self):
        model = self._build()
        assert isinstance(model.output_proj, nn.Linear)
        assert model.output_proj.in_features == 16
        assert model.output_proj.out_features == 3


class TestTransformerModelForward:
    def _build(self, n_features: int = 3, pred_len: int = 4) -> TransformerModel:
        return TransformerModel(
            n_features=n_features,
            pred_len=pred_len,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            activation="relu",
            context_len=8,
        )

    def test_when_called_without_targets_returns_pred_shape(self):
        model = self._build()
        x = torch.randn(2, 8, 3)
        out = model(x)
        assert out.shape == (2, 4, 3)

    def test_when_called_with_targets_returns_pred_shape(self):
        model = self._build()
        x = torch.randn(2, 8, 3)
        y = torch.randn(2, 4, 3)
        out = model(x, y=y)
        assert out.shape == (2, 4, 3)

    def test_when_pred_len_is_one_returns_single_step(self):
        model = self._build(pred_len=1)
        x = torch.randn(3, 8, 3)
        out = model(x)
        assert out.shape == (3, 1, 3)


class TestTransformerForecasterContract:
    def test_when_imported_is_subclass_of_encoder_decoder_forecaster(self):
        assert issubclass(TransformerForecaster, EncoderDecoderForecaster)

    def test_when_constructed_stores_context_and_pred_len(self):
        model = TransformerForecaster(context_len=10, pred_len=5)
        assert model.context_len == 10
        assert model.pred_len == 5

    def test_when_constructed_stores_transformer_hyperparameters(self):
        model = TransformerForecaster(
            d_model=128,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.2,
            activation="gelu",
            context_len=10,
            pred_len=5,
        )
        assert model.d_model == 128
        assert model.nhead == 8
        assert model.num_encoder_layers == 3
        assert model.num_decoder_layers == 3
        assert model.dim_feedforward == 512
        assert model.dropout == 0.2
        assert model.activation == "gelu"

    def test_when_constructed_modules_list_is_module(self):
        model = TransformerForecaster(context_len=4, pred_len=2)
        assert model._modules == ["module"]

    def test_when_model_type_accessed_routes_through_known_iterator(self):
        """Must be a key in base.py's self.datasets dict (currently rnn/cnn)."""
        from stockpy.preprocessing import TimeSeriesDataset

        model = TransformerForecaster(context_len=4, pred_len=2)
        assert model.model_type in {"rnn", "cnn"}
        # Sanity-check the route resolves.
        datasets = {"rnn": TimeSeriesDataset, "cnn": TimeSeriesDataset}
        assert datasets[model.model_type] is TimeSeriesDataset


class TestTransformerForecasterInitializeModule:
    def test_when_fit_called_module_is_instantiated(self):
        model = TransformerForecaster(
            context_len=8,
            pred_len=2,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
        )
        _fit_zero_epochs(model)
        assert hasattr(model, "module_")
        assert isinstance(model.module_, TransformerModel)

    def test_when_fit_called_n_features_in_propagates(self):
        model = TransformerForecaster(
            context_len=8,
            pred_len=2,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
        )
        _fit_zero_epochs(model, n_features=5)
        assert model.module_.input_proj.in_features == 5
        assert model.module_.output_proj.out_features == 5


class TestTransformerForecasterForward:
    def _ready(self) -> TransformerForecaster:
        model = TransformerForecaster(
            context_len=8,
            pred_len=2,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout=0.0,
        )
        _fit_zero_epochs(model)
        return model

    def test_when_called_returns_tensor_not_not_implemented(self):
        model = self._ready()
        x = torch.randn(1, 8, 3)
        out = model.forward(x)
        assert isinstance(out, torch.Tensor)

    def test_when_called_returns_pred_shape(self):
        model = self._ready()
        x = torch.randn(2, 8, 3)
        out = model.forward(x)
        assert out.shape == (2, 2, 3)

    def test_when_called_with_targets_returns_pred_shape(self):
        model = self._ready()
        x = torch.randn(2, 8, 3)
        y = torch.randn(2, 2, 3)
        out = model.forward(x, y=y)
        assert out.shape == (2, 2, 3)


class TestTransformerForecasterEndToEnd:
    def test_when_one_epoch_fit_history_records_epoch(self):
        model = TransformerForecaster(
            context_len=8,
            pred_len=2,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout=0.0,
        )
        X = np.random.randn(60, 3).astype(np.float32)
        y = np.random.randn(60, 3).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0, train_split=None)
        assert len(model.history) == 1

    def test_when_save_load_roundtrip_predictions_match(self, tmp_path):
        model = TransformerForecaster(
            context_len=8,
            pred_len=2,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout=0.0,
        )
        X = np.random.randn(60, 3).astype(np.float32)
        y = np.random.randn(60, 3).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0, train_split=None)

        f_params = str(tmp_path / "params.pt")
        model.save_params(f_params=f_params)

        X_test = torch.randn(4, 8, 3)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(4, 1))
        preds_before = model.predict(ds)

        model2 = TransformerForecaster(
            context_len=8,
            pred_len=2,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout=0.0,
        )
        model2.fit(X, y, epochs=0, verbose=0, train_split=None)
        model2.load_params(f_params=f_params)
        preds_after = model2.predict(ds)

        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)


# ----------------------------------------------------------------------------
# Issue #34 — canonical TestTransformerForecaster class.
# Mirrors test_lstm_forecaster.py / test_gru_forecaster.py / test_tcn_forecaster.py
# so the suite stays uniform across forecasters.
# ----------------------------------------------------------------------------

# Hyperparameters shared across the canonical tests (per issue #34 notes).
_HP = dict(
    d_model=16,
    nhead=2,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=32,
    dropout=0.1,
    context_len=10,
    pred_len=5,
)


def _make_forecaster(**overrides) -> TransformerForecaster:
    return TransformerForecaster(**{**_HP, **overrides})


def _state_dicts_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


class TestTransformerForecaster:
    """Canonical contract suite for ``TransformerForecaster``."""

    def test_when_instantiated_with_defaults_succeeds(self):
        model = TransformerForecaster()
        assert model.context_len == 20
        assert model.pred_len == 1
        assert model.d_model == 64
        assert model.nhead == 4

    def test_when_fit_5_epochs_loss_is_finite(self, synthetic_series):
        model = _make_forecaster(dropout=0.0)
        model.fit(
            synthetic_series, synthetic_series, epochs=5, verbose=0, train_split=None
        )
        assert len(model.history) == 5
        train_losses = model.history[:, "train_loss"]
        assert all(np.isfinite(loss) for loss in train_losses)

    def test_when_predict_called_returns_numpy_with_pred_shape(self, synthetic_series):
        model = _make_forecaster(dropout=0.0)
        model.fit(
            synthetic_series, synthetic_series, epochs=1, verbose=0, train_split=None
        )

        n_features = synthetic_series.shape[1]
        n_samples = 4
        X_test = torch.randn(n_samples, _HP["context_len"], n_features)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(n_samples, 1))
        preds = model.predict(ds)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (n_samples, _HP["pred_len"], n_features)

    def test_when_early_stopping_unreachable_threshold_training_ends_early(
        self, synthetic_series
    ):
        from stockpy.callbacks import EarlyStopping

        model = _make_forecaster(dropout=0.0)
        model.fit(
            synthetic_series,
            synthetic_series,
            epochs=20,
            verbose=0,
            train_split=None,
            callbacks=[
                EarlyStopping(
                    monitor="train_loss",
                    patience=2,
                    threshold=1000,
                    threshold_mode="abs",
                    lower_is_better=True,
                )
            ],
        )
        assert len(model.history) < 20

    def test_when_save_load_roundtrip_pt_state_dict_matches(
        self, synthetic_series, tmp_path
    ):
        model = _make_forecaster(dropout=0.0)
        model.fit(
            synthetic_series, synthetic_series, epochs=1, verbose=0, train_split=None
        )

        f_params = str(tmp_path / "params.pt")
        model.save_params(f_params=f_params)

        model2 = _make_forecaster(dropout=0.0)
        model2.fit(
            synthetic_series, synthetic_series, epochs=0, verbose=0, train_split=None
        )
        model2.load_params(f_params=f_params)

        assert _state_dicts_equal(
            model.module_.state_dict(), model2.module_.state_dict()
        )

    def test_when_save_load_roundtrip_safetensors_state_dict_matches(
        self, synthetic_series, tmp_path
    ):
        model = _make_forecaster(dropout=0.0)
        model.fit(
            synthetic_series, synthetic_series, epochs=1, verbose=0, train_split=None
        )

        f_params = str(tmp_path / "params.safetensors")
        model.save_params(f_params=f_params, use_safetensors=True)

        model2 = _make_forecaster(dropout=0.0)
        model2.fit(
            synthetic_series, synthetic_series, epochs=0, verbose=0, train_split=None
        )
        model2.device = None
        model2.load_params(f_params=f_params, use_safetensors=True)

        assert _state_dicts_equal(
            model.module_.state_dict(), model2.module_.state_dict()
        )

    def test_when_forward_with_teacher_forcing_returns_pred_shape(
        self, synthetic_series
    ):
        model = _make_forecaster(dropout=0.0)
        model.fit(
            synthetic_series, synthetic_series, epochs=0, verbose=0, train_split=None
        )

        n_features = synthetic_series.shape[1]
        x_batch = torch.randn(2, _HP["context_len"], n_features)
        y_batch = torch.randn(2, _HP["pred_len"], n_features)
        out = model.forward(x_batch, y=y_batch)
        assert out.shape == (2, _HP["pred_len"], n_features)

    def test_when_module_built_encoder_is_transformer_encoder(self, synthetic_series):
        model = _make_forecaster(dropout=0.0)
        model.fit(
            synthetic_series, synthetic_series, epochs=0, verbose=0, train_split=None
        )

        assert hasattr(model.module_, "encoder")
        assert isinstance(model.module_.encoder, nn.TransformerEncoder)

    def test_when_future_target_perturbed_earlier_predictions_unchanged(
        self, synthetic_series
    ):
        """Causal mask: position t cannot attend to positions > t."""
        torch.manual_seed(0)
        model = _make_forecaster(dropout=0.0)
        model.fit(
            synthetic_series, synthetic_series, epochs=1, verbose=0, train_split=None
        )
        model._set_training(False)

        n_features = synthetic_series.shape[1]
        x = torch.randn(1, _HP["context_len"], n_features)
        y_a = torch.randn(1, _HP["pred_len"], n_features)
        y_b = y_a.clone()
        y_b[:, -1, :] = y_b[:, -1, :] + 100.0  # perturb only the last step

        with torch.no_grad():
            out_a = model.forward(x, y=y_a)
            out_b = model.forward(x, y=y_b)

        # Earlier positions must be unchanged by perturbing the future target.
        assert torch.allclose(out_a[:, :-1, :], out_b[:, :-1, :], atol=1e-5)
