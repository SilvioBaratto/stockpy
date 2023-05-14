import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Union, Tuple, List
import torch
from sklearn.preprocessing import StandardScaler
from ..config import Config as cfg

class BaseStockDataset(torch.utils.data.Dataset):
    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor = None,
                 task: str = None
                 ):

        self.X = X
        self.y = y
        self.task = task

    def __len__(self):
        return self.X.shape[0]

    @property
    def input_size(self):
        return self.X.shape[1]

    @property
    def output_size(self):
        if self.task == 'regression':
            return self.y.shape[1]
        elif self.task == 'classification':
            return len(torch.unique(self.y))
