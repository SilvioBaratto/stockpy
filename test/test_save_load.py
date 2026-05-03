"""Tests for safetensors save/load round-trip on encoder-decoder forecasters.

Coverage for issue #38: ``BaseEstimator.load_params`` must coerce
``self.device`` (a ``torch.device``) to ``str`` before passing it to
``safetensors.safe_open``.
"""

import numpy as np
import torch

from stockpy.forecasters import LSTMForecaster


class TestSafetensorsRoundtrip:
    """Round-trip save → load → predict via safetensors with no workarounds."""

    def _build_and_fit(self):
        model = LSTMForecaster(
            context_len=10, pred_len=5, rnn_size=8, hidden_size=8, num_layers=1
        )
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0, train_split=None)
        return model, X, y

    def test_safetensors_roundtrip_cpu(self, tmp_path):
        model, X, y = self._build_and_fit()

        f_params = str(tmp_path / "params.safetensors")
        model.save_params(f_params=f_params, use_safetensors=True)

        X_test = torch.randn(5, 10, 3)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(5, 1))
        preds_before = model.predict(ds)

        model2 = LSTMForecaster(
            context_len=10, pred_len=5, rnn_size=8, hidden_size=8, num_layers=1
        )
        model2.fit(X, y, epochs=0, verbose=0, train_split=None)
        model2.load_params(f_params=f_params, use_safetensors=True)

        preds_after = model2.predict(ds)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

    def test_device_remains_torch_device_after_load(self, tmp_path):
        model, X, y = self._build_and_fit()

        f_params = str(tmp_path / "params.safetensors")
        model.save_params(f_params=f_params, use_safetensors=True)

        model2 = LSTMForecaster(
            context_len=10, pred_len=5, rnn_size=8, hidden_size=8, num_layers=1
        )
        model2.fit(X, y, epochs=0, verbose=0, train_split=None)
        device_before = model2.device
        model2.load_params(f_params=f_params, use_safetensors=True)

        assert isinstance(model2.device, torch.device)
        assert model2.device == device_before
