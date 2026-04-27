import numpy as np
import pytest
import torch

from stockpy.base import EncoderDecoderForecaster
from stockpy.preprocessing import TimeSeriesDataset


class TestConftestFixtures:
    """Tests that verify the conftest fixtures are properly set up."""

    def test_synthetic_series_fixture_exists(self, synthetic_series):
        assert isinstance(synthetic_series, np.ndarray)
        assert synthetic_series.ndim == 2
        assert synthetic_series.shape[1] == 3  # 3 features
        assert synthetic_series.dtype == np.float32

    def test_synthetic_series_has_reasonable_range(self, synthetic_series):
        # Sine waves with small noise should stay roughly in [-2, 2]
        assert synthetic_series.min() > -3.0
        assert synthetic_series.max() < 3.0

    def test_synthetic_series_train_val_test_shapes(self, synthetic_series):
        total_len = synthetic_series.shape[0]
        # Default split is 70/15/15
        assert total_len == 500

    def test_ts_dataset_fixture(self, ts_dataset):
        assert isinstance(ts_dataset, TimeSeriesDataset)
        assert ts_dataset.context_len == 20
        assert ts_dataset.pred_len == 5
        x, y = ts_dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (20, 3)
        assert y.shape == (5, 3)

    def test_mock_forecaster_fixture(self, mock_forecaster):
        assert isinstance(mock_forecaster, EncoderDecoderForecaster)
        assert mock_forecaster.context_len == 20
        assert mock_forecaster.pred_len == 5

    def test_mock_forecaster_predict_shape(self, mock_forecaster, synthetic_series):
        # Use a small batch of context windows
        X = synthetic_series[:30].reshape(1, 30, 3)
        y_pred = mock_forecaster.predict(X)
        assert y_pred.shape == (1, 5, 3)

    def test_train_val_test_split_shapes(self, train_series, val_series, test_series):
        assert isinstance(train_series, np.ndarray)
        assert isinstance(val_series, np.ndarray)
        assert isinstance(test_series, np.ndarray)

        total = train_series.shape[0] + val_series.shape[0] + test_series.shape[0]
        # Slight difference because of rounding in split percentages
        assert total <= 500
        assert total >= 498

        assert train_series.shape[1] == 3
        assert val_series.shape[1] == 3
        assert test_series.shape[1] == 3

    def test_train_val_test_are_deterministic(
        self, train_series, val_series, test_series
    ):
        # Because of fixed seed, these should always be the same
        assert train_series[0, 0] == pytest.approx(1.1077305, rel=1e-5)
        assert train_series[1, 1] == pytest.approx(-1.1217345, rel=1e-5)

    def test_synthetic_series_configurable(self):
        # The fixture itself is not parameterized, but we can verify
        # the helper function works with different params
        from stockpy.preprocessing._synthetic import make_synthetic_series

        series = make_synthetic_series(length=100, n_features=2, noise_std=0.05)
        assert series.shape == (100, 2)
        assert series.dtype == np.float32

    def test_mock_forecaster_with_callbacks(self, mock_forecaster):
        from stockpy.callbacks import Callback
        cb = Callback()
        # Should be able to attach callbacks
        mock_forecaster.callbacks = [cb]
        assert len(mock_forecaster.callbacks) == 1
