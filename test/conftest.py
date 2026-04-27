import numpy as np
import pytest

from stockpy.base import EncoderDecoderForecaster
from stockpy.preprocessing import TimeSeriesDataset
from stockpy.preprocessing._synthetic import make_synthetic_series


@pytest.fixture
def synthetic_series():
    return make_synthetic_series()


@pytest.fixture
def train_series(synthetic_series):
    split_idx = int(0.7 * len(synthetic_series))
    return synthetic_series[:split_idx]


@pytest.fixture
def val_series(synthetic_series):
    train_end = int(0.7 * len(synthetic_series))
    val_end = int(0.85 * len(synthetic_series))
    return synthetic_series[train_end:val_end]


@pytest.fixture
def test_series(synthetic_series):
    split_idx = int(0.85 * len(synthetic_series))
    return synthetic_series[split_idx:]


@pytest.fixture
def ts_dataset(synthetic_series):
    return TimeSeriesDataset(
        synthetic_series,
        context_len=20,
        pred_len=5,
    )


class MockForecaster(EncoderDecoderForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = True
        self.history = None
        self.callbacks = []
        self.initialized_ = True

    def predict(self, X, predict_nonlinearity="auto"):
        if hasattr(X, "shape"):
            n_samples = X.shape[0]
        else:
            n_samples = 1
        return np.zeros((n_samples, self.pred_len, 3), dtype=np.float32)

    def state_dict(self):
        if hasattr(self, "module_") and self.module_ is not None:
            return self.module_.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if hasattr(self, "module_") and self.module_ is not None:
            self.module_.load_state_dict(state_dict)

    def save_params(
        self,
        f_params=None,
        f_optimizer=None,
        f_history=None,
        f_module=None,
        use_safetensors=False,
        **kwargs,
    ):
        import os
        import torch
        if f_module is not None:
            f_params = f_module
        if f_params is not None:
            os.makedirs(os.path.dirname(f_params), exist_ok=True)
            if use_safetensors:
                from safetensors.torch import save_file
                save_file(self.module_.state_dict(), f_params)
            else:
                torch.save(self.module_.state_dict(), f_params)
        if f_optimizer is not None:
            os.makedirs(os.path.dirname(f_optimizer), exist_ok=True)
            torch.save(self.optimizer_.state_dict(), f_optimizer)
        if f_history is not None:
            os.makedirs(os.path.dirname(f_history), exist_ok=True)
            import json
            with open(f_history, "w") as f:
                json.dump([], f)


@pytest.fixture
def mock_forecaster():
    return MockForecaster(context_len=20, pred_len=5)
