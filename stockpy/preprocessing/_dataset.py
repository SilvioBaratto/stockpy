import numpy as np
import torch
import torch.utils.data

from stockpy.utils import multi_indexing
from stockpy.preprocessing import StockpyDataset

def unpack_data(data):
    """Unpack data returned by the net's iterator into a 2-tuple.

    If the wrong number of items is returned, raise a helpful error
    message.

    """
    # Note: This function cannot detect it when a user only returns 1
    # item that is exactly of length 2 (e.g. because the batch size is
    # 2). In that case, the item will be erroneously split into X and
    # y.
    X, y = data

    return X, y

class StockDatasetFFNN(StockpyDataset):
    """
    Class for Stock Dataset used for Feedforward Neural Networks (FFNNs).

    This class extends the BaseStockDataset class and implements the __getitem__ method for FFNNs.

    Attributes:
        Inherits all attributes from the BaseStockDataset class.
    """

    def __getitem__(self, i):
        """
        Retrieve the i-th item from the dataset.

        Parameters
        ----------
        i : int
            The index of the item to be retrieved.

        Returns
        -------
        tuple
            The i-th input data and target (if available).

        """
        # Extract the data and target for the specified index.
        X, y = self.X, self.y

        # If X is a pandas NDFrame, reshape its values.
        if self.X_is_ndframe:
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        # Extract the i-th data and target using proper indexing methods.
        Xi = multi_indexing(X, i, self.X_indexing)
        yi = multi_indexing(y, i, self.y_indexing)

        return self.transform(Xi, yi)
    
class StockDatasetRNN(StockpyDataset):

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
        Retrieve the i-th item from the dataset.

        Parameters
        ----------
        i : int
            The index of the item to be retrieved.

        Returns
        -------
        tuple
            The i-th input data and target (if available).

        """
        # Extract the data and target for the specified index.
        X, y = self.X, self.y

        # If X is a pandas NDFrame, reshape its values.
        if self.X_is_ndframe:
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        if i >= self.seq_len - 1:
            i_start = i - self.seq_len + 1
            X = self.X[i_start:(i + 1), :]

        else:
            padding_shape = (self.seq_len - i - 1, self.X.shape[1])
            padding = np.zeros(padding_shape, dtype=np.float32)
            X = self.X[0:(i + 1), :]
            X = np.concatenate((padding, X), axis=0)

        # Now X contains either a sequence from the dataset or a zero-padded sequence
        # Use multi_indexing to get the data at index i (or slice)
        Xi = multi_indexing(X, slice(0, self.seq_len), self.X_indexing)

        # Use multi_indexing to get the target at index i
        yi = multi_indexing(y, i, self.y_indexing)

        # Transform the indexed data and return
        return self.transform(Xi, yi)
    
class StockDatasetCNN(StockpyDataset):
    """
    Class for Stock Dataset used for Convolutional Neural Networks (CNNs).

    This class extends the BaseStockDataset class and implements the __getitem__ method for CNNs.

    Attributes:
        Inherits all attributes from the BaseStockDataset class.
    """

    def __getitem__(self, i):
        """
        Retrieve the i-th item from the dataset.

        Parameters
        ----------
        i : int
            The index of the item to be retrieved.

        Returns
        -------
        tuple
            The i-th input data and target (if available).

        """
        # Extract the data and target for the specified index.
        X, y = self.X, self.y

        # If X is a pandas NDFrame, reshape its values.
        if self.X_is_ndframe:
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        # if tensor unsqueeze if numpy reshape
        if isinstance(X, torch.Tensor):
            X = X.unsqueeze(1).float()
        else:
            X = np.expand_dims(X, axis=1).astype(np.float32)

        # Extract the i-th data and target using proper indexing methods.
        Xi = multi_indexing(X, i, self.X_indexing)
        yi = multi_indexing(y, i, self.y_indexing)

        return self.transform(Xi, yi)

class StockDatasetSeq2Seq(StockpyDataset):

    def __init__(
            self,
            X,
            y,
            length,
            seq_len
    ):
        
        super(StockDatasetSeq2Seq, self).__init__(X, y, length)

        self.seq_len = seq_len

    def __getitem__(self, i):
        """
        Retrieve the i-th item from the dataset.

        Parameters
        ----------
        i : int
            The index of the item to be retrieved.

        Returns
        -------
        tuple
            The i-th input data and target (if available).

        """
        # Extract the data and target for the specified index.
        X, y = self.X, self.y

        # If X is a pandas NDFrame, reshape its values.
        if self.X_is_ndframe:
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        is_torch_tensor = isinstance(X, torch.Tensor)

        if i >= self.seq_len - 1:
            # If the index i is greater than or equal to the sequence length minus one 
            # (defined in cfg.training.seq_len), it selects a sequence of data 
            # from the dataset self.X starting from i_start to i (both inclusive). 
            # The sequence length is defined in the configuration.

            i_start = i - self.seq_len + 1
            X = multi_indexing(self.X, slice(i_start, i+1), self.X_indexing)
        else:
            # If i is less than the sequence length minus one, it creates a zero padding for 
            # the missing data to ensure that the input always has the same shape. 
            # This is done by creating a tensor of zeros with the appropriate shape using torch.zeros(), 
            # then concatenating this padding with the actual data using torch.cat(). 
            # This is a common practice in machine learning when working with sequences of varying length, 
            # especially for RNNs.

            if is_torch_tensor:
                padding = torch.zeros((self.seq_len - i - 1, self.X.shape[1]), dtype=torch.float32)
            else:
                padding = np.zeros((self.seq_len - i - 1, self.X.shape[1]), dtype=np.float32)
                
            X = multi_indexing(self.X, slice(0, i + 1), self.X_indexing)

            if is_torch_tensor:
                X = torch.cat((padding, X), 0)
            else:
                X = np.concatenate((padding, X), axis=0)

        # Handle y
        if y is not None:
            if i >= self.seq_len - 1:
                i_start = i - self.seq_len + 1
                y = multi_indexing(self.y, slice(i_start, i + 1), self.y_indexing)
            else:
                # Create a padding tensor for y
                padding_y_shape = (self.seq_len - i - 1, self.y.shape[1])
                padding_y = torch.zeros(padding_y_shape, dtype=torch.float32) if is_torch_tensor else np.zeros(padding_y_shape, dtype=np.float32)
                
                # Extract the relevant slice of y
                y = multi_indexing(self.y, slice(0, i + 1), self.y_indexing)

                # Concatenate the padding tensor and y
                y = torch.cat((padding_y, y), 0) if is_torch_tensor else np.concatenate((padding_y, y), axis=0)


        return self.transform(X, y)