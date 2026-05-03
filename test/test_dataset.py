import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from stockpy.preprocessing import TimeSeriesDataset


class TestTimeSeriesDataset:
    """Tests for TimeSeriesDataset with context_len and pred_len windows."""

    def test_constructor_accepts_context_len_pred_len_stride(self):
        X = np.arange(100).reshape(50, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=10, pred_len=5, stride=1)
        assert ds.context_len == 10
        assert ds.pred_len == 5
        assert ds.stride == 1

    def test_getitem_returns_tuple_of_tensors(self):
        X = np.arange(24).reshape(12, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=4, pred_len=3)
        x_context, y_target = ds[0]
        assert isinstance(x_context, torch.Tensor)
        assert isinstance(y_target, torch.Tensor)

    def test_context_shape(self):
        X = np.arange(100).reshape(50, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=10, pred_len=5)
        x_context, _ = ds[0]
        assert x_context.shape == (10, 2)

    def test_target_shape(self):
        X = np.arange(100).reshape(50, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=10, pred_len=5)
        _, y_target = ds[0]
        assert y_target.shape == (5, 2)

    def test_correct_window_content(self):
        X = np.arange(30).reshape(15, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=4, pred_len=3)
        x_context, y_target = ds[2]

        expected_context = X[2:6]
        expected_target = X[6:9]

        np.testing.assert_array_equal(x_context.numpy(), expected_context)
        np.testing.assert_array_equal(y_target.numpy(), expected_target)

    def test_length_with_stride_one(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=3, pred_len=2)
        assert len(ds) == 10 - 3 - 2 + 1
        assert len(ds) == 6

    def test_length_with_stride_two(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=3, pred_len=2, stride=2)
        expected_len = (10 - 3 - 2) // 2 + 1
        assert len(ds) == expected_len
        assert len(ds) == 3

    def test_last_window(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=3, pred_len=2)
        last_idx = len(ds) - 1
        x_context, y_target = ds[last_idx]

        expected_context = X[5:8]
        expected_target = X[8:10]

        np.testing.assert_array_equal(x_context.numpy(), expected_context)
        np.testing.assert_array_equal(y_target.numpy(), expected_target)

    def test_works_with_dataloader(self):
        X = np.arange(100).reshape(50, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=10, pred_len=5)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        x_batch, y_batch = batch

        assert x_batch.shape == (4, 10, 2)
        assert y_batch.shape == (4, 5, 2)

    def test_dataloader_multiple_batches(self):
        X = np.arange(40).reshape(20, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=4, pred_len=3)
        loader = DataLoader(ds, batch_size=3, shuffle=False)

        batches = list(loader)
        assert len(batches) == 5  # 14 windows / 3 = 4 full + 1 partial

        # First batch
        x0, y0 = batches[0]
        assert x0.shape == (3, 4, 2)
        assert y0.shape == (3, 3, 2)

        # Last batch (remainder)
        x_last, y_last = batches[-1]
        assert x_last.shape[0] == 2  # 14 % 3 = 2

    def test_series_too_short_raises(self):
        X = np.arange(6).reshape(3, 2).astype(np.float32)
        with pytest.raises(ValueError, match="too short"):
            TimeSeriesDataset(X, context_len=3, pred_len=2)

    def test_exact_fit_length(self):
        X = np.arange(10).reshape(5, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=3, pred_len=2)
        assert len(ds) == 1
        x_context, y_target = ds[0]
        np.testing.assert_array_equal(x_context.numpy(), X[0:3])
        np.testing.assert_array_equal(y_target.numpy(), X[3:5])

    def test_1d_input_raises_or_works(self):
        X = np.arange(10).astype(np.float32)
        with pytest.raises(ValueError, match="2-dimensional"):
            TimeSeriesDataset(X, context_len=3, pred_len=2)

    def test_with_torch_tensor_input(self):
        X = torch.arange(20).reshape(10, 2).float()
        ds = TimeSeriesDataset(X, context_len=3, pred_len=2)
        x_context, y_target = ds[0]
        assert isinstance(x_context, torch.Tensor)
        assert x_context.shape == (3, 2)
        assert y_target.shape == (2, 2)

    def test_stride_greater_than_one_windows(self):
        X = np.arange(40).reshape(20, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=4, pred_len=2, stride=3)

        x0, y0 = ds[0]
        np.testing.assert_array_equal(x0.numpy(), X[0:4])
        np.testing.assert_array_equal(y0.numpy(), X[4:6])

        x1, y1 = ds[1]
        np.testing.assert_array_equal(x1.numpy(), X[3:7])
        np.testing.assert_array_equal(y1.numpy(), X[7:9])

    def test_default_stride_is_one(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=3, pred_len=2)
        assert ds.stride == 1

    def test_separate_y_target(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        y = np.arange(30).reshape(10, 3).astype(np.float32)
        ds = TimeSeriesDataset(X, y=y, context_len=3, pred_len=2)

        x_context, y_target = ds[0]
        assert x_context.shape == (3, 2)
        assert y_target.shape == (2, 3)

        np.testing.assert_array_equal(x_context.numpy(), X[0:3])
        np.testing.assert_array_equal(y_target.numpy(), y[3:5])

    def test_mismatched_x_y_lengths_raises(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        y = np.arange(16).reshape(8, 2).astype(np.float32)
        with pytest.raises(ValueError, match="inconsistent lengths"):
            TimeSeriesDataset(X, y=y, context_len=3, pred_len=2)

    def test_negative_stride_raises(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        with pytest.raises(ValueError, match="stride"):
            TimeSeriesDataset(X, context_len=3, pred_len=2, stride=0)

    def test_context_len_zero_raises(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        with pytest.raises(ValueError, match="context_len"):
            TimeSeriesDataset(X, context_len=0, pred_len=2)

    def test_pred_len_zero_raises(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        with pytest.raises(ValueError, match="pred_len"):
            TimeSeriesDataset(X, context_len=3, pred_len=0)

    def test_empty_series_raises(self):
        X = np.empty((0, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="too short"):
            TimeSeriesDataset(X, context_len=3, pred_len=2)

    def test_single_feature(self):
        X = np.arange(5).reshape(5, 1).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=3, pred_len=2)
        x_context, y_target = ds[0]
        assert x_context.shape == (3, 1)
        assert y_target.shape == (2, 1)
        np.testing.assert_array_equal(x_context.numpy(), X[0:3])
        np.testing.assert_array_equal(y_target.numpy(), X[3:5])

    def test_negative_index_raises(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=3, pred_len=2)
        with pytest.raises(IndexError, match="out of bounds"):
            ds[-1]

    def test_out_of_bounds_index_raises(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        ds = TimeSeriesDataset(X, context_len=3, pred_len=2)
        with pytest.raises(IndexError, match="out of bounds"):
            ds[100]

    def test_stride_one_equals_no_stride(self):
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        ds_default = TimeSeriesDataset(X, context_len=3, pred_len=2)
        ds_explicit = TimeSeriesDataset(X, context_len=3, pred_len=2, stride=1)
        assert len(ds_default) == len(ds_explicit)
        for i in range(len(ds_default)):
            x_def, y_def = ds_default[i]
            x_exp, y_exp = ds_explicit[i]
            np.testing.assert_array_equal(x_def.numpy(), x_exp.numpy())
            np.testing.assert_array_equal(y_def.numpy(), y_exp.numpy())

    def test_with_synthetic_series(self, synthetic_series):
        ds = TimeSeriesDataset(synthetic_series, context_len=20, pred_len=5)
        x_context, y_target = ds[0]
        assert x_context.shape == (20, 3)
        assert y_target.shape == (5, 3)
        assert isinstance(x_context, torch.Tensor)
        assert isinstance(y_target, torch.Tensor)


class TestShortSeriesUpfrontValidation:
    """Issue #41: short-series guard must fire at ``fit`` entry, before any
    training machinery is built."""

    def test_fit_raises_before_training_on_short_series(self):
        from stockpy.forecasters import LSTMForecaster

        X = np.zeros((10, 3), dtype=np.float32)
        model = LSTMForecaster(
            context_len=30, pred_len=5, rnn_size=8, hidden_size=8, num_layers=1
        )

        with pytest.raises(
            ValueError,
            match=r"context_len.*pred_len|pred_len.*context_len",
        ):
            model.fit(X, y=X, epochs=1, verbose=0, train_split=None)

        assert getattr(model, "initialized_", False) is False
        history = getattr(model, "history", None)
        assert history is None or len(history) == 0

    def test_error_message_reports_actual_length(self):
        from stockpy.forecasters import LSTMForecaster

        X = np.zeros((7, 2), dtype=np.float32)
        model = LSTMForecaster(
            context_len=10, pred_len=4, rnn_size=4, hidden_size=4, num_layers=1
        )

        with pytest.raises(ValueError, match=r"\b7\b"):
            model.fit(X, y=X, epochs=1, verbose=0, train_split=None)

    def test_dataset_constructor_guard_still_fires(self):
        X = np.zeros((5, 2), dtype=np.float32)
        with pytest.raises(ValueError, match=r"context_len.*pred_len|too short"):
            TimeSeriesDataset(X, context_len=10, pred_len=3)

    def test_guard_skipped_when_X_has_no_len(self):
        from stockpy.forecasters import LSTMForecaster

        class _NoLenDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                yield torch.zeros(10, 3), torch.zeros(5, 3)

        model = LSTMForecaster(
            context_len=10, pred_len=5, rnn_size=4, hidden_size=4, num_layers=1
        )
        try:
            model.fit(_NoLenDataset(), epochs=0, verbose=0, train_split=None)
        except ValueError as exc:
            msg = str(exc)
            assert not (
                "context_len" in msg and "pred_len" in msg and "Series length" in msg
            )
        except Exception:
            pass
