import numpy as np
import pytest
import torch

from stockpy.preprocessing import StandardScalerTransform, DifferenceTransform


class TestStandardScalerTransform:
    """Tests for StandardScalerTransform (z-score per feature)."""

    def test_fit_computes_mean_and_std(self):
        data = np.arange(12).reshape(4, 3).astype(np.float32)
        scaler = StandardScalerTransform()
        scaler.fit(data)

        expected_mean = np.mean(data, axis=0)
        expected_std = np.std(data, axis=0, ddof=0)

        np.testing.assert_array_almost_equal(scaler.mean_, expected_mean)
        np.testing.assert_array_almost_equal(scaler.std_, expected_std)

    def test_standard_scaler_returns_numpy_for_numpy_input(self):
        data = np.arange(12).reshape(4, 3).astype(np.float32)
        scaler = StandardScalerTransform()
        scaler.fit(data)
        transformed = scaler.transform(data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.dtype == np.float32

    def test_standard_scaler_returns_tensor_for_tensor_input(self):
        data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        scaler = StandardScalerTransform()
        scaler.fit(data.numpy())
        transformed = scaler.transform(data)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.dtype == torch.float32

    def test_inverse_transform_returns_numpy_for_numpy_input(self):
        data = np.arange(12).reshape(4, 3).astype(np.float32)
        scaler = StandardScalerTransform()
        scaler.fit(data)
        scaled = scaler.transform(data)
        recovered = scaler.inverse_transform(scaled)

        assert isinstance(recovered, np.ndarray)
        np.testing.assert_array_almost_equal(recovered, data, decimal=5)

    def test_inverse_transform_returns_tensor_for_tensor_input(self):
        data = np.arange(12).reshape(4, 3).astype(np.float32)
        scaler = StandardScalerTransform()
        scaler.fit(data)
        scaled = torch.from_numpy(scaler.transform(data)).float()
        recovered = scaler.inverse_transform(scaled)

        assert isinstance(recovered, torch.Tensor)

    def test_transform_produces_correct_z_scores(self):
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        scaler = StandardScalerTransform()
        scaler.fit(data)
        transformed = scaler.transform(data)

        expected = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
        np.testing.assert_array_almost_equal(transformed, expected, decimal=5)

    def test_transform_on_unseen_data(self):
        train = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        test = np.array([[7, 8], [9, 10]], dtype=np.float32)

        scaler = StandardScalerTransform()
        scaler.fit(train)
        transformed = scaler.transform(test)

        expected = (test - train.mean(axis=0)) / train.std(axis=0, ddof=0)
        np.testing.assert_array_almost_equal(transformed, expected, decimal=5)

    def test_inverse_transform_recovers_original(self):
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        scaler = StandardScalerTransform()
        scaler.fit(data)
        transformed = scaler.transform(data)
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(recovered, data, decimal=5)

    def test_fit_transform_combined(self):
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        scaler = StandardScalerTransform()
        transformed = scaler.fit_transform(data)

        assert isinstance(transformed, np.ndarray)
        expected = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
        np.testing.assert_array_almost_equal(transformed, expected, decimal=5)

    def test_constant_feature_raises(self):
        data = np.array([[1, 2], [1, 4], [1, 6]], dtype=np.float32)
        scaler = StandardScalerTransform()
        with pytest.raises(ValueError, match="zero standard deviation"):
            scaler.fit(data)

    def test_transform_before_fit_raises(self):
        data = np.ones((3, 2), dtype=np.float32)
        scaler = StandardScalerTransform()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            scaler.transform(data)

    def test_inverse_transform_before_fit_raises(self):
        data = torch.ones(3, 2)
        scaler = StandardScalerTransform()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            scaler.inverse_transform(data)

    def test_1d_input_raises(self):
        data = np.arange(5).astype(np.float32)
        scaler = StandardScalerTransform()
        with pytest.raises(ValueError, match="2-dimensional"):
            scaler.fit(data)

    def test_single_sample_raises(self):
        data = np.array([[1, 2]], dtype=np.float32)
        scaler = StandardScalerTransform()
        with pytest.raises(ValueError, match="at least 2 samples"):
            scaler.fit(data)


class TestDifferenceTransform:
    """Tests for DifferenceTransform (first-order differencing)."""

    def test_transform_returns_torch_tensor(self):
        data = np.arange(10).reshape(5, 2).astype(np.float32)
        diff = DifferenceTransform()
        result = diff.transform(data)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_first_order_difference(self):
        data = np.array([[1, 10], [3, 12], [6, 15], [10, 19]], dtype=np.float32)
        diff = DifferenceTransform()
        result = diff.transform(data)

        expected = np.diff(data, axis=0).astype(np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_transform_reduces_length_by_one(self):
        data = np.arange(20).reshape(10, 2).astype(np.float32)
        diff = DifferenceTransform()
        result = diff.transform(data)

        assert result.shape[0] == data.shape[0] - 1
        assert result.shape[1] == data.shape[1]

    def test_order_two_difference(self):
        data = np.array([[1, 10], [3, 12], [6, 15], [10, 19]], dtype=np.float32)
        diff = DifferenceTransform(order=2)
        result = diff.transform(data)

        expected = np.diff(data, n=2, axis=0).astype(np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_single_row_raises(self):
        data = np.array([[1, 2]], dtype=np.float32)
        diff = DifferenceTransform()
        with pytest.raises(ValueError, match="at least 2 rows"):
            diff.transform(data)

    def test_order_greater_than_rows_raises(self):
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        diff = DifferenceTransform(order=3)
        with pytest.raises(ValueError, match="order"):
            diff.transform(data)

    def test_fit_is_noop(self):
        data = np.arange(10).reshape(5, 2).astype(np.float32)
        diff = DifferenceTransform()
        diff.fit(data)  # should not raise or do anything harmful
        assert True

    def test_fit_transform_equivalent(self):
        data = np.arange(10).reshape(5, 2).astype(np.float32)
        diff = DifferenceTransform()
        result = diff.fit_transform(data)
        expected = diff.transform(data)
        np.testing.assert_array_equal(result.numpy(), expected.numpy())

    def test_no_inverse_transform(self):
        data = np.arange(6).reshape(3, 2).astype(np.float32)
        diff = DifferenceTransform()
        result = diff.transform(data)
        with pytest.raises(NotImplementedError):
            diff.inverse_transform(result)

    def test_1d_input_raises(self):
        data = np.arange(5).astype(np.float32)
        diff = DifferenceTransform()
        with pytest.raises(ValueError, match="2-dimensional"):
            diff.transform(data)

    def test_zero_order_raises(self):
        data = np.ones((3, 2), dtype=np.float32)
        diff = DifferenceTransform(order=0)
        with pytest.raises(ValueError, match="order"):
            diff.transform(data)

    def test_negative_order_raises(self):
        data = np.ones((3, 2), dtype=np.float32)
        diff = DifferenceTransform(order=-1)
        with pytest.raises(ValueError, match="order"):
            diff.transform(data)
