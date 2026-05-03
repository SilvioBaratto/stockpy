import numpy as np
import torch

__all__ = ["StandardScalerTransform", "DifferenceTransform"]


class StandardScalerTransform:
    """
    Z-score (standard) scaler for time-series features.

    Computes mean and standard deviation per feature on the training
    data and applies (x - mean) / std.

    Parameters
    ----------
    None

    Attributes
    ----------
    mean_ : np.ndarray, shape (n_features,)
        Feature-wise mean learned during ``fit``.
    std_ : np.ndarray, shape (n_features,)
        Feature-wise standard deviation learned during ``fit``.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def _check_is_fitted(self):
        """Raise if fit has not been called."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError(
                "StandardScalerTransform has not been fitted yet. "
                "Call fit(data) before transform or inverse_transform."
            )

    def _validate_input(self, data):
        """Ensure data is a 2D numpy array with at least 2 samples."""
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(
                "Input must be 2-dimensional with shape (n_samples, n_features). "
                f"Got shape with {data.ndim} dimension(s)."
            )
        if data.shape[0] < 2:
            raise ValueError(
                "Input must contain at least 2 samples to compute statistics."
            )
        return data

    def fit(self, data):
        """
        Compute mean and standard deviation per feature.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Training data used to compute statistics.

        Returns
        -------
        StandardScalerTransform
            The fitted scaler.
        """
        data = self._validate_input(data)

        self.mean_ = data.mean(axis=0)
        self.std_ = data.std(axis=0, ddof=0)

        if np.any(self.std_ == 0):
            raise ValueError(
                "One or more features have zero standard deviation. "
                "Cannot scale constant features."
            )

        return self

    def transform(self, data):
        """
        Standardize data using precomputed statistics.

        Parameters
        ----------
        data : array-like or torch.Tensor, shape (n_samples, n_features)
            Data to standardize.

        Returns
        -------
        np.ndarray or torch.Tensor
            Standardized data. The return type mirrors the input: a
            ``torch.Tensor`` is returned when ``data`` is a tensor,
            otherwise a float32 numpy array.
        """
        self._check_is_fitted()
        is_tensor = isinstance(data, torch.Tensor)
        array = data.detach().cpu().numpy() if is_tensor else np.asarray(data)
        if array.ndim != 2:
            raise ValueError(
                "Input must be 2-dimensional with shape (n_samples, n_features). "
                f"Got shape with {array.ndim} dimension(s)."
            )

        scaled = ((array - self.mean_) / self.std_).astype(np.float32)
        return torch.from_numpy(scaled) if is_tensor else scaled

    def inverse_transform(self, data):
        """
        Reverse standardization.

        Parameters
        ----------
        data : array-like or torch.Tensor, shape (n_samples, n_features)
            Scaled data to invert.

        Returns
        -------
        np.ndarray or torch.Tensor
            Original-scale data. The return type mirrors the input.
        """
        self._check_is_fitted()
        is_tensor = isinstance(data, torch.Tensor)
        array = data.detach().cpu().numpy() if is_tensor else np.asarray(data)

        original = (array * self.std_ + self.mean_).astype(np.float32)
        return torch.from_numpy(original) if is_tensor else original

    def fit_transform(self, data):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        data : array-like or torch.Tensor, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        np.ndarray or torch.Tensor
            Standardized data; return type mirrors the input.
        """
        return self.fit(data).transform(data)


class DifferenceTransform:
    """
    First-order (or n-th order) differencing for time series.

    Computes discrete differences along the time (sample) axis,
    producing a stationary approximation of the series.

    Parameters
    ----------
    order : int, default=1
        Number of times to apply differencing.

    Attributes
    ----------
    order : int
        Differencing order.
    """

    def __init__(self, order=1):
        if not isinstance(order, int):
            raise TypeError("order must be an integer.")
        self.order = order

    def _validate_input(self, data):
        """Ensure data is a 2D numpy array."""
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(
                "Input must be 2-dimensional with shape (n_samples, n_features). "
                f"Got shape with {data.ndim} dimension(s)."
            )
        if data.shape[0] < 2:
            raise ValueError(
                "Input must contain at least 2 rows to compute differences."
            )
        return data

    def fit(self, data):
        """
        No-op for DifferenceTransform (stateless).

        Parameters
        ----------
        data : array-like
            Ignored.

        Returns
        -------
        DifferenceTransform
            The instance.
        """
        return self

    def transform(self, data):
        """
        Apply n-th order differencing along the time axis.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Time series data.

        Returns
        -------
        torch.Tensor
            Differenced data as a float32 tensor with shape
            ``(n_samples - order, n_features)``.
        """
        data = self._validate_input(data)

        if self.order <= 0:
            raise ValueError(f"order must be a positive integer, got {self.order}.")
        if self.order >= data.shape[0]:
            raise ValueError(
                f"order ({self.order}) must be less than the number of rows "
                f"({data.shape[0]})."
            )

        diff = np.diff(data, n=self.order, axis=0)
        return torch.from_numpy(diff).float()

    def inverse_transform(self, data):
        """
        Inverse transform is not defined for differencing.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "inverse_transform is not defined for DifferenceTransform."
        )

    def fit_transform(self, data):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Time series data.

        Returns
        -------
        torch.Tensor
            Differenced data as a float32 tensor.
        """
        return self.fit(data).transform(data)
