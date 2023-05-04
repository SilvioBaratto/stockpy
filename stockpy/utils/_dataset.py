import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Union, Tuple, List
from ._model import Model
import torch
from sklearn.preprocessing import StandardScaler
from ..config import Config as cfg

class TradingStockDatasetRNN(torch.utils.data.Dataset):
    def __init__(self, 
                X: Union[np.ndarray, pd.core.frame.DataFrame], 
                y: Union[np.ndarray, pd.core.frame.DataFrame] = None,
                ):

        self.X = torch.tensor(X.values).float()
        if y is not None:
            self.y = self.y = torch.tensor(y.values).reshape(-1, 1 if len(y.shape) == 1 \
                                                         or y.shape[1] == 1 \
                                                         else y.shape[1]).float()
        else:
            self.y = None

    def __len__(self):
        return self.X.shape[0]

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
    
    @property
    def input_size(self):
        return self.X.shape[1]
    
    @property
    def output_size(self):
        return self.y.shape[1]
    
class TradingStockDatasetFFNN(torch.utils.data.Dataset):
    def __init__(self, 
                X: Union[np.ndarray, pd.core.frame.DataFrame], 
                y: Union[np.ndarray, pd.core.frame.DataFrame] = None,
                ):

        self.X = torch.tensor(X.values).float()
        if y is not None:
            self.y = self.y = torch.tensor(y.values).reshape(-1, 1 if len(y.shape) == 1 \
                                                         or y.shape[1] == 1 \
                                                         else y.shape[1]).float()
        else:
            self.y = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if self.y is None:
            return self.X[i]
        else:
            return self.X[i], self.y[i]
    
    @property
    def input_size(self):
        return self.X.shape[1]
    
    @property
    def output_size(self):
        return self.y.shape[1]
    
class TradingStockDatasetCNN(torch.utils.data.Dataset):
    def __init__(self, 
                X: Union[np.ndarray, pd.core.frame.DataFrame], 
                y: Union[np.ndarray, pd.core.frame.DataFrame] = None,
                ):

        self.X = torch.tensor(X.values).unsqueeze(1).float()  # Add channel dimension
        if y is not None:
            self.y = torch.tensor(y.values).reshape(-1, 1 if len(y.shape) == 1 \
                                                         or y.shape[1] == 1 \
                                                         else y.shape[1]).float()
        else:
            self.y = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if self.y is None:
            return self.X[i]
        else:
            return self.X[i], self.y[i]
    
    @property
    def input_size(self):
        return self.X.shape[2]  # Changed from shape[1] to shape[2]
    
    @property
    def output_size(self):
        return self.y.shape[1]
    
class ClassifierStockDatasetRNN(torch.utils.data.Dataset):
    def __init__(self,
                 X: Union[np.ndarray, pd.core.frame.DataFrame],
                 y: Union[np.ndarray, pd.core.frame.DataFrame]
                 ):
        self.X = torch.tensor(X.values).float()
        self.y = torch.tensor(y.squeeze().values).long() - 1  # Convert y to a 1D tensor of long (integer) type

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= cfg.training.sequence_length - 1:
            i_start = i - cfg.training.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = torch.zeros((cfg.training.sequence_length - i - 1, self.X.size(1)), dtype=torch.float32)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

    @property
    def input_size(self):
        return self.X.shape[1]

    @property
    def output_size(self):
        return len(torch.unique(self.y))
    
class ClassifierStockDatasetFFNN(torch.utils.data.Dataset):
    def __init__(self,
                 X: Union[np.ndarray, pd.core.frame.DataFrame],
                 y: Union[np.ndarray, pd.core.frame.DataFrame]
                 ):
        self.X = torch.tensor(X.values).float()
        self.y = torch.tensor(y.squeeze().values).long() - 1  # Convert y to a 1D tensor of long (integer) type

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    @property
    def input_size(self):
        return self.X.shape[1]

    @property
    def output_size(self):
        return len(torch.unique(self.y))
    
class ClassifierStockDatasetCNN(torch.utils.data.Dataset):
    def __init__(self,
                 X: Union[np.ndarray, pd.core.frame.DataFrame],
                 y: Union[np.ndarray, pd.core.frame.DataFrame]
                 ):
        self.X = torch.tensor(X.values).float().unsqueeze(1)  # Add channel dimension
        self.y = torch.tensor(y.squeeze().values).long() - 1  # Convert y to a 1D tensor of long (integer) type

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if self.y is None:
            return self.X[i]
        else:
            return self.X[i], self.y[i]

    @property
    def input_size(self):
        return self.X.shape[1:]

    @property
    def output_size(self):
        return len(torch.unique(self.y))
    
class GenStockDataset(torch.utils.data.Dataset):
    def __init__(self, 
                dataframe, 
                ):

        self.dataframe = dataframe

        x = dataframe.values
        self.X = torch.tensor(x).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= cfg.training.sequence_length - 1:
            i_start = i - cfg.training.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
            y = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(cfg.training.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
            y = self.X[0:(i + 1), :]
            y = torch.cat((padding, y), 0)

        return x, y
    
    @property
    def input_size(self):
        return self.X.shape[1]

    @property
    def output_size(self):
        return self.y.shape[1]