import torch

from stockpy.preprocessing import StockpyDataset
from stockpy.utils import multi_indexing

__all__ = ["TimeSeriesDataset", "unpack_data"]


def unpack_data(data):
    """
    Unpack data returned by the net's iterator into a 2-tuple.

    This function is designed to be used within a loop where data is
    being iterated, typically from a DataLoader in PyTorch. It expects
    each item of the iteration to be a 2-tuple or a 2-element list, which
    it then unpacks into the input data (features) and the target data
    (labels). If the data iterable doesn't contain exactly two elements,
    it is considered an error and raises an exception.

    Parameters
    ----------
    data : iterable
        An iterable that yields elements, which should be pairs (2-tuple or
        2-list) of features and labels.

    Returns
    -------
    tuple
        A 2-tuple where the first element is the unpacked features and the
        second element is the unpacked labels.

    Raises
    ------
    ValueError
        If `data` does not contain exactly two elements.

    Notes
    -----
    This function cannot detect it when a user only returns 1
    item that is exactly of length 2 (e.g., because the batch size is
    2). In that case, the item will be erroneously split into X and y.
    """
    if len(data) != 2:
        raise ValueError(
            f"Expected data to be a 2-tuple or 2-list, got {len(data)} elements instead."
        )

    X, y = data
    return X, y


class TimeSeriesDataset(StockpyDataset):
    """
    A dataset class for time-series forecasting with context and prediction windows.

    This class produces sliding windows over a multivariate time series,
    returning past-context and future-target pairs suitable for
    encoder-decoder forecasters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input time series data.
    y : array-like, shape (n_samples, n_targets) or None, optional
        The target time series data. If ``None``, the target is derived
        from ``X`` itself (auto-regressive forecasting).
    length : int, optional
        The total number of samples. Inferred from ``X`` if not given.
    context_len : int, default=20
        Number of past time steps to use as model input.
    pred_len : int, default=1
        Number of future time steps to predict.
    stride : int, default=1
        Step size between consecutive windows.

    Attributes
    ----------
    context_len : int
        Length of the context window.
    pred_len : int
        Length of the prediction window.
    stride : int
        Stride between consecutive windows.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(24).reshape(12, 2).astype(np.float32)
    >>> ds = TimeSeriesDataset(X, context_len=4, pred_len=3)
    >>> x, y = ds[0]
    >>> x.shape
    torch.Size([4, 2])
    >>> y.shape
    torch.Size([3, 2])
    """

    def __init__(
        self,
        X,
        y=None,
        length=None,
        context_len=20,
        pred_len=1,
        stride=1,
    ):
        super().__init__(X, y, length=length)

        if hasattr(self.X, "shape") and len(self.X.shape) != 2:
            raise ValueError(
                "X must be 2-dimensional with shape (time_steps, features). "
                f"Got shape with {len(self.X.shape)} dimension(s)."
            )

        if context_len <= 0:
            raise ValueError("context_len must be a positive integer")
        if pred_len <= 0:
            raise ValueError("pred_len must be a positive integer")
        if stride <= 0:
            raise ValueError("stride must be a positive integer")

        self.context_len = context_len
        self.pred_len = pred_len
        self.stride = stride

        series_len = self._len
        min_required = self.context_len + self.pred_len
        if series_len < min_required:
            raise ValueError(
                f"Series length ({series_len}) is too short for "
                f"context_len ({context_len}) + pred_len ({pred_len}). "
                f"Minimum required: {min_required}."
            )

        self._len = (series_len - min_required) // self.stride + 1

    def __getitem__(self, i):
        """
        Retrieve the i-th sliding window from the dataset.

        Parameters
        ----------
        i : int
            Index of the window.

        Returns
        -------
        tuple
            A 2-tuple ``(x_context, y_target)`` where both elements are
            ``torch.Tensor``.

            * ``x_context`` has shape ``(context_len, n_features)``.
            * ``y_target`` has shape ``(pred_len, n_features)`` when
              ``y`` is ``None``, or ``(pred_len, n_targets)`` when
              ``y`` is provided.
        """
        if i < 0 or i >= self._len:
            raise IndexError(
                f"Index {i} is out of bounds for dataset with length {self._len}"
            )

        start_idx = i * self.stride
        context_end = start_idx + self.context_len
        target_end = context_end + self.pred_len

        Xi = multi_indexing(self.X, slice(start_idx, context_end), self.X_indexing)

        if self.y is not None:
            yi = multi_indexing(self.y, slice(context_end, target_end), self.y_indexing)
        else:
            yi = multi_indexing(self.X, slice(context_end, target_end), self.X_indexing)

        Xi, yi = self.transform(Xi, yi)

        Xi = torch.as_tensor(Xi, dtype=torch.float32)
        yi = torch.as_tensor(yi, dtype=torch.float32)

        return Xi, yi
