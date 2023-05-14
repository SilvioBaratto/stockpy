import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Union, Tuple, List
import torch
from ._dataset import StockDatasetRNN
from ._dataset import StockDatasetFFNN
from ._dataset import StockDatasetCNN
from ..config import Config as cfg

class ZScoreNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data: torch.Tensor):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)
        # Ensure that the standard deviation is not zero
        self.std = torch.where(self.std != 0, self.std, torch.ones_like(self.std))

    def denormalize(self, data: torch.Tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError('Must fit normalizer before denormalizing data.')
        return data * self.std + self.mean

class StockScaler:
    
    def __init__(self):             
        self.X_normalizer = ZScoreNormalizer()
        self.y_normalizer = ZScoreNormalizer()

    def fit_transform(self, 
                      X_train: Union[np.ndarray, pd.core.frame.DataFrame],
                      y_train: Union[np.ndarray, pd.core.frame.DataFrame] = None,
                      task: str = None):
        
        X_train = torch.tensor(X_train.values).float()
        self.X_normalizer.fit(X_train)
        X_train = (X_train - self.X_normalizer.mean) / self.X_normalizer.std

        if y_train is not None and task == 'regression':
            y_train = torch.tensor(y_train.values).reshape(-1, 1 if len(y_train.shape) == 1 \
                                                            or y_train.shape[1] == 1 \
                                                            else y_train.shape[1]).float()
            self.y_normalizer.fit(y_train)
            y_train = (y_train - self.y_normalizer.mean) / self.y_normalizer.std

        elif task == 'classification':
            y_train = torch.tensor(y_train.values).squeeze().long() - 1

        else:
            y_train = None
        
        return X_train, y_train

    def transform(self, 
                  X_test: torch.Tensor):
        
        X_test = (X_test - self.X_normalizer.mean) / self.X_normalizer.std
        
        return X_test
    
    def inverse_transform(self,
                          y_pred: torch.Tensor):

        return y_pred * self.y_normalizer.std + self.y_normalizer.mean

class StockDataloader:
    def __init__(self,
                 X: Union[np.ndarray, pd.core.frame.DataFrame],
                 y: Union[np.ndarray, pd.core.frame.DataFrame] = None,
                 model_type: str = None,
                 task: str = None
                 ):
        
        self.scaler = StockScaler()
        self.task = task
        
        self.X_train, self.y_train = self.scaler.fit_transform(X, y, task)
        self.model_type = model_type

        self.datasets = {
            'rnn': StockDatasetRNN,
            'ffnn': StockDatasetFFNN,
            'cnn': StockDatasetCNN
        }

        self.dataset = self.datasets[self.model_type](self.X_train, self.y_train, self.task)

    def get_loader(self, 
                X: Union[np.ndarray, pd.core.frame.DataFrame] = None, 
                y: Union[np.ndarray, pd.core.frame.DataFrame] = None,
                mode: str = 'train'):
        
        if mode == 'train':
            start_idx = 0
            end_idx = int(0.8 * len(self.dataset))
            subset = torch.utils.data.Subset(self.dataset, range(start_idx, end_idx))
        
        elif mode == 'val':
            start_idx = int(0.8 * len(self.dataset))
            end_idx = len(self.dataset)
            subset = torch.utils.data.Subset(self.dataset, range(start_idx, end_idx))
        elif mode == 'test':
            X = torch.tensor(X.values).float()
            X_test = self.scaler.transform(X)
            if self.task == 'regression':
                subset = self.datasets[self.model_type](X=X_test, 
                                                        y=None, 
                                                        task=self.task)
            elif self.task == 'classification':
                # Ensure y is a tensor of longs, representing class labels
                y = torch.tensor(y.values).squeeze().long() - 1
                subset = self.datasets[self.model_type](X=X_test, 
                                                        y=y, 
                                                        task=self.task)
        else:
            raise ValueError("Invalid mode. Accepted modes: 'train', 'val' or 'test'")
        
        return DataLoader(subset, 
                        batch_size=cfg.training.batch_size,
                        shuffle=cfg.training.shuffle
                        )

    def inverse_transform_output(self, y_pred):
        return self.scaler.inverse_transform(y_pred)
