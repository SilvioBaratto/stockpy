import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Union, Tuple, List
import torch
from sklearn.preprocessing import StandardScaler
from ._base import BaseStockDataset
from ..config import Config as cfg

class StockDatasetRNN(BaseStockDataset):
    def __getitem__(self, i):
        if i >= cfg.training.sequence_length - 1:
            i_start = i - cfg.training.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = torch.zeros((cfg.training.sequence_length - i - 1, self.X.size(1)), dtype=torch.float32)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        if self.y is None:
            return x
        else:
            return x, self.y[i]


class StockDatasetFFNN(BaseStockDataset):
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
