import numpy as np
import pytest
import torch
import torch.nn as nn

from stockpy.forecasters import BiLSTMForecaster
from stockpy.base import EncoderDecoderForecaster
from stockpy.preprocessing import TimeSeriesDataset


class TestBiLSTMForecaster:
    """Comprehensive tests for BiLSTMForecaster encoder-decoder implementation."""

    def test_is_subclass_of_encoder_decoder_forecaster(self):
        assert issubclass(BiLSTMForecaster, EncoderDecoderForecaster)

    def test_instantiation_sets_context_and_pred_len(self):
        model = BiLSTMForecaster(context_len=10, pred_len=5)
        assert model.context_len == 10
        assert model.pred_len == 5

    def test_model_type_attribute(self):
        model = BiLSTMForecaster(context_len=4, pred_len=2)
        assert model.model_type == "rnn"

    def test_initialize_module_creates_module_(self):
        model = BiLSTMForecaster(context_len=4, pred_len=2, rnn_size=8, hidden_size=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        assert hasattr(model, "module_")
        assert isinstance(model.module_, nn.Module)

    def test_module_has_bidirectional_encoder_and_unidirectional_decoder(self):
        model = BiLSTMForecaster(context_len=4, pred_len=2, rnn_size=8, hidden_size=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        assert hasattr(model.module_, "encoder")
        assert hasattr(model.module_, "decoder")
        assert isinstance(model.module_.encoder, nn.LSTM)
        assert isinstance(model.module_.decoder, nn.LSTM)
        assert model.module_.encoder.bidirectional is True
        assert model.module_.decoder.bidirectional is False

    def test_decoder_hidden_size_matches_encoder_bidirectional_output(self):
        model = BiLSTMForecaster(context_len=4, pred_len=2, rnn_size=8, hidden_size=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        # Bidirectional encoder outputs 2 * rnn_size, decoder must match
        assert model.module_.decoder.hidden_size == 2 * model.rnn_size

    def test_forward_output_shape(self):
        model = BiLSTMForecaster(context_len=4, pred_len=2, rnn_size=8, hidden_size=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)

        x_batch = torch.randn(2, 4, 3)
        out = model.forward(x_batch)
        assert out.shape == (2, 2, 3)

    def test_forward_with_teacher_forcing_shape(self):
        model = BiLSTMForecaster(context_len=4, pred_len=2, rnn_size=8, hidden_size=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)

        x_batch = torch.randn(2, 4, 3)
        y_batch = torch.randn(2, 2, 3)
        out = model.forward(x_batch, y=y_batch)
        assert out.shape == (2, 2, 3)

    def test_teacher_forcing_changes_output(self):
        model = BiLSTMForecaster(context_len=4, pred_len=2, rnn_size=8, hidden_size=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=2, verbose=0, train_split=None)

        x_batch = torch.randn(2, 4, 3)
        y_batch = torch.randn(2, 2, 3)
        out_no_tf = model.forward(x_batch)
        out_tf = model.forward(x_batch, y=y_batch)
        assert not torch.allclose(out_no_tf, out_tf, atol=1e-6)

    def test_predict_returns_correct_shape(self):
        model = BiLSTMForecaster(
            context_len=10, pred_len=5, rnn_size=16, hidden_size=16, num_layers=1
        )
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0, train_split=None)

        X_test = torch.randn(5, 10, 3)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(5, 1))
        preds = model.predict(ds)
        assert preds.shape == (5, 5, 3)

    def test_fit_on_synthetic_data(self):
        from stockpy.preprocessing._synthetic import make_synthetic_series

        series = make_synthetic_series(length=200, n_features=3)
        X = series
        y = series
        model = BiLSTMForecaster(
            context_len=20, pred_len=5, rnn_size=16, hidden_size=16, num_layers=1
        )
        model.fit(X, y, epochs=2, verbose=0, train_split=None)
        assert model.initialized_
        assert len(model.history) == 2

    def test_save_load_roundtrip(self, tmp_path):
        model = BiLSTMForecaster(
            context_len=10, pred_len=5, rnn_size=8, hidden_size=8, num_layers=1
        )
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0, train_split=None)

        f_params = str(tmp_path / "params.pt")
        model.save_params(f_params=f_params)

        X_test = torch.randn(5, 10, 3)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(5, 1))
        preds_before = model.predict(ds)

        model2 = BiLSTMForecaster(
            context_len=10, pred_len=5, rnn_size=8, hidden_size=8, num_layers=1
        )
        model2.fit(X, y, epochs=0, verbose=0, train_split=None)
        model2.load_params(f_params=f_params)

        preds_after = model2.predict(ds)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

    def test_set_training_mode(self):
        model = BiLSTMForecaster(context_len=4, pred_len=2, rnn_size=8, hidden_size=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        model._set_training(False)
        assert model.module_.training is False
        model._set_training(True)
        assert model.module_.training is True
        model._set_training(False)
        assert model.module_.training is False

    def test_dropout_applies_in_training(self):
        model = BiLSTMForecaster(
            context_len=4, pred_len=2, rnn_size=8, hidden_size=8, dropout=0.5
        )
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)

        x_batch = torch.randn(2, 4, 3)
        model._set_training(True)
        out1 = model.forward(x_batch)
        out2 = model.forward(x_batch)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_forward_does_not_raise_not_implemented(self):
        model = BiLSTMForecaster(context_len=4, pred_len=2, rnn_size=8, hidden_size=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        x_batch = torch.randn(1, 4, 3)
        out = model.forward(x_batch)
        assert isinstance(out, torch.Tensor)
