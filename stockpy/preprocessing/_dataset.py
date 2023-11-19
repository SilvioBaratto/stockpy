import numpy as np
import torch
import torch.utils.data

from stockpy.utils import multi_indexing
from stockpy.preprocessing import StockpyDataset

__all__ = ['StockDatasetFFNN', 'StockDatasetRNN', 'StockDatasetCNN', 'unpack_data']
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

class StockDatasetFFNN(StockpyDataset):
    """
    A dataset class for stock data suited for Feedforward Neural Networks (FFNNs).

    This class extends `StockpyDataset` and customizes the `__getitem__` method to
    provide data in a format that is directly consumable by FFNNs.

    Attributes
    ----------
    X : various types
        Input data in formats supported by `StockpyDataset`.
    y : various types or None
        Target data in formats supported by `StockpyDataset`, or None if not applicable.
    length : int
        The length of the dataset as defined in `StockpyDataset`.
    X_indexing : str or callable
        The indexing strategy used for `X`, inherited from `StockpyDataset`.
    y_indexing : str or callable
        The indexing strategy used for `y`, inherited from `StockpyDataset`.
    X_is_ndframe : bool
        Boolean flag indicating whether `X` is a pandas NDFrame, inherited from `StockpyDataset`.
    """

    def __getitem__(self, i):
        """
        Retrieve the i-th sample from the dataset, formatted for FFNN input.

        The method handles indexing into the dataset's `X` and `y` attributes,
        reshapes data if necessary, and applies any required transformations.

        Parameters
        ----------
        i : int
            The index of the sample to be retrieved from the dataset.

        Returns
        -------
        tuple
            A 2-tuple where the first element is the feature vector (`Xi`) and the
            second element is the target (`yi`), if `y` is provided. If `X` is a
            pandas NDFrame, it reshapes its values to comply with FFNN input requirements.

        Raises
        ------
        IndexError
            If the index `i` is out of bounds of the dataset.
        """
        # Extract the data and target for the specified index.
        X, y = self.X, self.y

        # If X is a pandas NDFrame, reshape its values.
        if self.X_is_ndframe:
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        # Extract the i-th data and target using the indexing strategy.
        Xi = multi_indexing(X, i, self.X_indexing)
        yi = multi_indexing(y, i, self.y_indexing)

        # Apply transformations and return the tuple.
        return self.transform(Xi, yi)
    
class StockDatasetRNN(StockpyDataset):
    """
    A dataset class specifically tailored for Recurrent Neural Networks (RNNs).

    This class processes stock data into sequences suitable for temporal models.
    It inherits from `StockpyDataset` and adds sequence processing capabilities.

    Attributes
    ----------
    seq_len : int
        The length of the sequences to be generated for RNN inputs.
    X : array-like, shape (n_samples, n_features)
        The input data containing stock features.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        The target values. If `None`, the dataset has only inputs (unsupervised).
    length : int
        The total number of samples in the dataset.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data containing stock features.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        The target values. If `None`, the dataset has only inputs (unsupervised).
    length : int
        The total number of samples in the dataset.
    seq_len : int
        The length of the data sequences to be used by the RNN.
    """

    def __init__(
            self,
            X,
            y,
            length,
            seq_len
    ):

        super(StockDatasetRNN, self).__init__(X, y, length)

        self.seq_len = seq_len

    def __getitem__(self, i):
        """
        Retrieve the i-th time sequence from the dataset.

        If the requested index does not allow for a full sequence, the method
        generates a zero-padded sequence. The padding is prepended to ensure
        that the output always has the shape `(seq_len, n_features)`.

        Parameters
        ----------
        i : int
            The index of the time sequence to be retrieved from the dataset.

        Returns
        -------
        tuple
            A tuple where the first element is a 2D array representing the feature sequence
            for the RNN (`Xi`), and the second element is the corresponding target (`yi`).

        Raises
        ------
        IndexError
            If the index `i` is out of bounds considering the sequence length.
        """
        # Extract the data and target for the specified index.
        X, y = self.X, self.y

        if i < 0 or i >= len(X):
            raise IndexError(f"Index {i} is out of bounds for dataset with length {len(X)}")

        is_numpy = isinstance(X, np.ndarray)
        is_torch = isinstance(X, torch.Tensor)

        if is_numpy:
            zero_func = np.zeros
            concat_func = np.concatenate
        elif is_torch:
            zero_func = torch.zeros
            concat_func = torch.cat
        else:
            raise TypeError("Unsupported data type for X. Expected numpy.ndarray or torch.Tensor")

        if i >= self.seq_len - 1:
            X_seq = X[i - self.seq_len + 1:i + 1, :]
        else:
            padding_shape = (self.seq_len - (i + 1), X.shape[1])
            padding = zero_func(padding_shape, dtype=X.dtype)
            if is_torch:
                padding = padding.to(X.device)  # Ensure padding is on the same device as X
            X_seq = concat_func((padding, X[:i + 1, :]), axis=0)

        assert X_seq.shape[0] == self.seq_len, f"Sequence length mismatch: expected {self.seq_len}, got {X_seq.shape[0]}"

        # Get the feature sequence for the RNN
        Xi = multi_indexing(X_seq, slice(0, self.seq_len), self.X_indexing)

        # Get the corresponding target
        yi = multi_indexing(y, i, self.y_indexing)

        # Apply any transformations if necessary (e.g., normalization)
        return self.transform(Xi, yi)
    
class StockDatasetCNN(StockpyDataset):
    """
    A dataset class designed for use with Convolutional Neural Networks (CNNs).

    This class processes stock market data, preparing it for convolutional operations.
    It inherits from `StockpyDataset` and customizes the `__getitem__` method to 
    accommodate the input requirements of CNNs by reshaping the data appropriately.

    Attributes
    ----------
    Inherits all attributes from the BaseStockDataset class, with additional processing
    to prepare the data for CNNs.
    """

    def __getitem__(self, i):
        """
        Retrieve the i-th sample from the dataset, reshaping it for use with a CNN.

        For CNNs, the input X needs to have a specific shape, often adding a channel 
        dimension, which this method ensures.

        Parameters
        ----------
        i : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple of the reshaped input data suitable for CNN processing (`Xi`)
            and its corresponding target (`yi`), if available.

        Raises
        ------
        IndexError
            If the index `i` is out of bounds for the dataset.
        """
        # Extract the data and target for the specified index.
        X, y = self.X, self.y

        # If X is a pandas NDFrame, reshape its values.
        if self.X_is_ndframe:
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        # Adjust the shape of X to have a channel dimension for CNN processing.
        if isinstance(X, torch.Tensor):
            X = X.unsqueeze(1).float()
        else:
            X = np.expand_dims(X, axis=1).astype(np.float32)

        # Extract the i-th data and target using proper indexing methods.
        Xi = multi_indexing(X, i, self.X_indexing)
        yi = multi_indexing(y, i, self.y_indexing)

        # Transform the indexed data and return.
        return self.transform(Xi, yi)

# class StockDatasetSeq2Seq(StockpyDataset):

#     def __init__(
#             self,
#             X,
#             y,
#             length,
#             seq_len
#     ):
        
#         super(StockDatasetSeq2Seq, self).__init__(X, y, length)

#         self.seq_len = seq_len

#     def __getitem__(self, i):
#         """
#         Retrieve the i-th item from the dataset.

#         Parameters
#         ----------
#         i : int
#             The index of the item to be retrieved.

#         Returns
#         -------
#         tuple
#             The i-th input data and target (if available).

#         """
#         # Extract the data and target for the specified index.
#         X, y = self.X, self.y

#         # If X is a pandas NDFrame, reshape its values.
#         if self.X_is_ndframe:
#             X = {k: X[k].values.reshape(-1, 1) for k in X}

#         is_torch_tensor = isinstance(X, torch.Tensor)

#         if i >= self.seq_len - 1:
#             # If the index i is greater than or equal to the sequence length minus one 
#             # (defined in cfg.training.seq_len), it selects a sequence of data 
#             # from the dataset self.X starting from i_start to i (both inclusive). 
#             # The sequence length is defined in the configuration.

#             i_start = i - self.seq_len + 1
#             X = multi_indexing(self.X, slice(i_start, i+1), self.X_indexing)
#         else:
#             # If i is less than the sequence length minus one, it creates a zero padding for 
#             # the missing data to ensure that the input always has the same shape. 
#             # This is done by creating a tensor of zeros with the appropriate shape using torch.zeros(), 
#             # then concatenating this padding with the actual data using torch.cat(). 
#             # This is a common practice in machine learning when working with sequences of varying length, 
#             # especially for RNNs.

#             if is_torch_tensor:
#                 padding = torch.zeros((self.seq_len - i - 1, self.X.shape[1]), dtype=torch.float32)
#             else:
#                 padding = np.zeros((self.seq_len - i - 1, self.X.shape[1]), dtype=np.float32)
                
#             X = multi_indexing(self.X, slice(0, i + 1), self.X_indexing)

#             if is_torch_tensor:
#                 X = torch.cat((padding, X), 0)
#             else:
#                 X = np.concatenate((padding, X), axis=0)

#         # Handle y
#         if y is not None:
#             if i >= self.seq_len - 1:
#                 i_start = i - self.seq_len + 1
#                 y = multi_indexing(self.y, slice(i_start, i + 1), self.y_indexing)
#             else:
#                 # Create a padding tensor for y
#                 padding_y_shape = (self.seq_len - i - 1, self.y.shape[1])
#                 padding_y = torch.zeros(padding_y_shape, dtype=torch.float32) if is_torch_tensor else np.zeros(padding_y_shape, dtype=np.float32)
                
#                 # Extract the relevant slice of y
#                 y = multi_indexing(self.y, slice(0, i + 1), self.y_indexing)

#                 # Concatenate the padding tensor and y
#                 y = torch.cat((padding_y, y), 0) if is_torch_tensor else np.concatenate((padding_y, y), axis=0)


#         return self.transform(X, y)