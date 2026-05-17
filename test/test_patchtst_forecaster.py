"""Tests for PatchTSTForecaster.

Coverage for issue #43: encoder-only Transformer with channel-independent
patching and a direct multi-step head.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from stockpy.base import EncoderDecoderForecaster
from stockpy.forecasters import PatchTSTForecaster


def _fit_zero_epochs(model: PatchTSTForecaster, n_features: int = 3) -> None:
    X = np.random.randn(120, n_features).astype(np.float32)
    y = np.random.randn(120, n_features).astype(np.float32)
    model.fit(X, y, epochs=0, verbose=0, train_split=None)


class TestPatchTSTForecasterContract:
    def test_is_subclass_of_encoder_decoder_forecaster(self):
        assert issubclass(PatchTSTForecaster, EncoderDecoderForecaster)

    def test_instantiation_sets_context_and_pred_len(self):
        model = PatchTSTForecaster(context_len=10, pred_len=5, patch_len=10)
        assert model.context_len == 10
        assert model.pred_len == 5

    def test_model_type_attribute(self):
        model = PatchTSTForecaster(context_len=4, pred_len=2, patch_len=4)
        assert model.model_type == "rnn"

    def test_initialize_module_creates_module_(self):
        model = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
        )
        _fit_zero_epochs(model)
        assert hasattr(model, "module_")
        assert isinstance(model.module_, nn.Module)

    def test_forward_output_shape(self):
        model = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        _fit_zero_epochs(model)
        x = torch.randn(2, 16, 3)
        out = model.forward(x)
        assert out.shape == (2, 4, 3)

    def test_forward_ignores_y(self):
        model = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        _fit_zero_epochs(model)
        x = torch.randn(2, 16, 3)
        y = torch.randn(2, 4, 3)
        out_with_y = model.forward(x, y=y)
        out_without_y = model.forward(x)
        assert torch.allclose(out_with_y, out_without_y)

    def test_predict_returns_correct_shape(self):
        model = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0, train_split=None)

        X_test = torch.randn(5, 16, 3)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(5, 1))
        preds = model.predict(ds)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (5, 4, 3)


class TestPatchTSTArchitecture:
    def test_patch_count_correct(self):
        model = PatchTSTForecaster(
            context_len=96,
            pred_len=24,
            patch_len=16,
            stride=8,
            d_model=128,
            nhead=16,
            num_layers=1,
            dim_feedforward=512,
        )
        _fit_zero_epochs(model)
        # With context_len=96, patch_len=16, stride=8:
        # (96-16)=80, divisible by 8, no padding, N=11
        expected_num_patches = 11
        assert model.module_.num_patches == expected_num_patches

    def test_revin_round_trip(self):
        torch.manual_seed(0)
        np.random.seed(0)
        # Constant series — after RevIN normalisation and denormalisation,
        # the output should equal the input.
        X = np.ones((120, 3), dtype=np.float32)
        y = np.ones((120, 3), dtype=np.float32)
        model = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            revin=True,
            revin_affine=False,
        )
        model.fit(X, y, epochs=5, verbose=0, train_split=None)

        X_test = torch.ones(1, 16, 3)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(1, 1))
        preds = model.predict(ds)
        # preds shape: (1, 4, 3) — all ones
        np.testing.assert_allclose(preds, np.ones((1, 4, 3)), atol=1e-4)

    def test_channel_independence_shared_weights(self):
        # Build two models differing only in n_features.
        model1 = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            revin=True,
            revin_affine=True,
        )
        model8 = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            revin=True,
            revin_affine=True,
        )
        _fit_zero_epochs(model1, n_features=1)
        _fit_zero_epochs(model8, n_features=8)

        params1 = sum(p.numel() for p in model1.module_.parameters())
        params8 = sum(p.numel() for p in model8.module_.parameters())
        # Difference should be exactly the RevIN affine params: 2 * n_features
        assert params8 - params1 == 2 * 8 - 2 * 1

    def test_invalid_ctor_raises(self):
        with pytest.raises(ValueError, match="context_len"):
            PatchTSTForecaster(context_len=8, patch_len=16)

        with pytest.raises(ValueError, match="d_model"):
            PatchTSTForecaster(d_model=127, nhead=16)

        with pytest.raises(ValueError, match="patch_len"):
            PatchTSTForecaster(patch_len=0)

        with pytest.raises(ValueError, match="stride"):
            PatchTSTForecaster(stride=0)


class TestPatchTSTForecasterEndToEnd:
    def test_fit_on_synthetic_data(self):
        from stockpy.preprocessing._synthetic import make_synthetic_series

        torch.manual_seed(42)
        np.random.seed(42)
        series = make_synthetic_series(length=200, n_features=3)
        model = PatchTSTForecaster(
            context_len=20,
            pred_len=5,
            patch_len=10,
            stride=5,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )
        model.fit(
            series,
            series,
            epochs=5,
            verbose=0,
            train_split=None,
            optimizer=torch.optim.Adam,
            lr=1e-3,
        )
        assert len(model.history) == 5
        train_losses = [model.history[i, "train_loss"] for i in range(5)]
        # Loss should decrease monotonically over first 5 epochs
        for i in range(1, 5):
            assert train_losses[i] < train_losses[i - 1]

    def test_save_load_roundtrip(self, tmp_path):
        model = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0, train_split=None)

        f_params = str(tmp_path / "params.pt")
        model.save_params(f_params=f_params)

        X_test = torch.randn(5, 16, 3)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(5, 1))
        model.module_.eval()
        preds_before = model.predict(ds)

        model2 = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        model2.fit(X, y, epochs=0, verbose=0, train_split=None)
        model2.load_params(f_params=f_params)
        model2.module_.eval()
        preds_after = model2.predict(ds)

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-6)

    def test_set_training_mode(self):
        model = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
        )
        _fit_zero_epochs(model)
        model._set_training(False)
        assert model.module_.training is False
        model._set_training(True)
        assert model.module_.training is True

    def test_dropout_applies_in_training(self):
        model = PatchTSTForecaster(
            context_len=16,
            pred_len=4,
            patch_len=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.5,
        )
        _fit_zero_epochs(model)
        x = torch.randn(2, 16, 3)
        model._set_training(True)
        out1 = model.forward(x)
        out2 = model.forward(x)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_when_constructed_stores_hyperparameters(self):
        model = PatchTSTForecaster(
            patch_len=8,
            stride=4,
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            revin=False,
            revin_affine=False,
            context_len=32,
            pred_len=8,
        )
        assert model.patch_len == 8
        assert model.stride == 4
        assert model.d_model == 64
        assert model.nhead == 8
        assert model.num_layers == 2
        assert model.dim_feedforward == 256
        assert model.dropout == 0.1
        assert model.activation == "gelu"
        assert model.revin is False
        assert model.revin_affine is False
        assert model.context_len == 32
        assert model.pred_len == 8

    def test_when_constructed_modules_list_is_module(self):
        model = PatchTSTForecaster(context_len=4, pred_len=2, patch_len=4)
        assert model._modules == ["module"]
