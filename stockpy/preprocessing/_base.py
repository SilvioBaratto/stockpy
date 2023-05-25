import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Union, Tuple, List
import torch
from sklearn.preprocessing import StandardScaler
from ..config import Config as cfg

class BaseStockDataset(torch.utils.data.Dataset):
    """
    Base class for Stock Dataset.

    This class represents a base dataset for stock data. It extends from PyTorch's Dataset class.

    Parameters:
        X (torch.Tensor): The input data tensor.
        y (torch.Tensor, optional): The target data tensor. Default is None.
        task (str, optional): The type of task. Could be either 'regression' or 'classification'. Default is None.

    Attributes:
        X (torch.Tensor): The input data tensor.
        y (torch.Tensor): The target data tensor.
        task (str): The type of task. Could be either 'regression' or 'classification'.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor = None, task: str = None):
        """
        Initializes the BaseStockDataset instance.
        
        Args:
            X (torch.Tensor): The input data tensor.
            y (torch.Tensor, optional): The target data tensor. Default is None.
            task (str, optional): The type of task. Could be either 'regression' or 'classification'. Default is None.
        """
        self.X = X
        self.y = y
        self.task = task

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.X.shape[0]

    @property
    def input_size(self):
        """
        The size of the input data.

        Returns:
            int: The number of features in the input data.
        """
        return self.X.shape[1]

    @property
    def output_size(self):
        """
        The size of the output data.

        If the task is 'regression', it returns the number of targets in the output data.
        If the task is 'classification', it returns the number of unique classes in the output data.

        Returns:
            int: The size of the output data.
        """
        if self.task == 'regression':
            return self.y.shape[1]
        elif self.task == 'classification':
            return len(torch.unique(self.y))
