import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Union, Tuple, List
import torch
from sklearn.preprocessing import StandardScaler
from ._base import BaseStockDataset
from ..config import Config as cfg

__doc__ = """
This module contains the dataset classes for the stockpy library. The dataset classes are used to create 
the PyTorch DataLoader objects that are used to train the models. The dataset classes are also used to 
create the train, validation and test datasets.
"""

class StockDatasetRNN(BaseStockDataset):
    def __getitem__(self, i):
        if i >= cfg.training.sequence_length - 1:
            # If the index i is greater than or equal to the sequence length minus one 
            # (defined in cfg.training.sequence_length), it selects a sequence of data 
            # from the dataset self.X starting from i_start to i (both inclusive). 
            # The sequence length is defined in the configuration.

            i_start = i - cfg.training.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            # If i is less than the sequence length minus one, it creates a zero padding for 
            # the missing data to ensure that the input always has the same shape. 
            # This is done by creating a tensor of zeros with the appropriate shape using torch.zeros(), 
            # then concatenating this padding with the actual data using torch.cat(). 
            # This is a common practice in machine learning when working with sequences of varying length, 
            # especially for RNNs.

            padding = torch.zeros((cfg.training.sequence_length - i - 1, self.X.size(1)), dtype=torch.float32)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        if self.y is None:
            return x
        else:
            return x, self.y[i]


class StockDatasetFFNN(BaseStockDataset):
    # returns the ith element from the dataset self.X. If the targets self.y exist, it returns a 
    # tuple of the input data and the corresponding target. If they don't exist, it only returns 
    # the input data.

    def __getitem__(self, i):
        x = self.X[i, :]

        if self.y is None:
            return x
        else:
            return x, self.y[i]


class StockDatasetCNN(BaseStockDataset):
    def __init__(self,
                 X: Union[np.ndarray, pd.core.frame.DataFrame],
                 y: Union[np.ndarray, pd.core.frame.DataFrame] = None,
                 task: str = 'regression'
                 ):
        super().__init__(X, y, task)
        # add an extra dimension to X using unsqueeze(1), which is necessary because 
        # CNNs expect input data to have a specific shape (including a channel dimension).

        self.X = X.unsqueeze(1).float()  # Add channel dimension

    def __getitem__(self, i):
        x = self.X[i, :, :]

        if self.y is None:
            return x
        else:
            return x, self.y[i]

    @property
    def input_size(self):
        return self.X.shape[2]  # Changed from shape[1] to shape[2]
