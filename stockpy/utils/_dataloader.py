import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Union, Tuple, List
from ._model import Model
import torch
from sklearn.preprocessing import StandardScaler
from ._dataset import TradingStockDatasetCNN
from ._dataset import TradingStockDatasetRNN
from ._dataset import TradingStockDatasetFFNN
from ._dataset import ClassifierStockDatasetCNN
from ._dataset import ClassifierStockDatasetRNN
from ._dataset import ClassifierStockDatasetFFNN
from ..config import Config as cfg

class StockDataset():
    
    def __init__(self, 
                 X: Union[np.ndarray, pd.core.frame.DataFrame],
                 y: Union[np.ndarray, pd.core.frame.DataFrame] = None,
                 scale_y: bool = True,
                 ):
        
        super().__init__()
        
        self.X_scaler = self._initScaler()
        self.y_scaler = self._initScaler() if y is not None else None
        self.X = self._fit_transform(X, self.X_scaler)
        self.y = self._fit_transform(y, self.y_scaler) if y is not None and scale_y else y

    def getDl(self, category, model_class):
        return self._initDl(X=self.X, 
                            y=self.y, 
                            category=category,
                            model_class=model_class)
    
    def getTestDl(self, category, model_class, X, y=None):
        return self._initDl(X=X, 
                            y=y, 
                            category=category,
                            model_class=model_class)
    
    def getValDl(self, category, model_class):
        X_val = self.X.iloc[int(len(self.X) * 0.8):]
        y_val = self.y.iloc[int(len(self.y) * 0.8):] if self.y is not None else None
        return self._initDl(X=X_val, y=y_val, category=category, model_class=model_class)

    def _initScaler(self):
        return StandardScaler()

    def _fit_transform(self, data, scaler):
        if data is None:
            return None
        
        # Convert the Series to a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        if len(data.columns) == 1:
            data_np = data.to_numpy().reshape(-1, 1)
        else:
            data_np = data.to_numpy()
        
        data_scaled = scaler.fit_transform(data_np)
        return pd.DataFrame(data_scaled,
                            columns=data.columns,
                            index=data.index)

    def _inverse_transform(self, data, scaler):
        if data is None or scaler is None:
            return None

        # Convert the Series to a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()

        if len(data.columns) == 1:
            data_np = data.to_numpy().reshape(-1, 1)
        else:
            data_np = data.to_numpy()

        data_inverse = scaler.inverse_transform(data_np)
        return pd.DataFrame(data_inverse,
                            columns=data.columns,
                            index=data.index)
    
    def _std_y(self):
        return self.y_scaler.scale_
    
    def _mean_y(self):
        return self.y_scaler.mean_
    
    def _get_x_scaler(self):
        return self.X_scaler
    
    def _get_y_scaler(self):
        return self.y_scaler

    def _initDl(self,
                X: Union[np.ndarray, pd.core.frame.DataFrame],
                y: Union[np.ndarray, pd.core.frame.DataFrame],
                category,
                model_class) -> torch.utils.data.DataLoader:
                
        if category == "regressor":
            dataloader = {
                "rnn": TradingStockDatasetRNN,
                "cnn": TradingStockDatasetCNN,
                "ffnn": TradingStockDatasetFFNN
            }
            return DataLoader(dataloader[model_class](X, y),
                              batch_size=cfg.training.batch_size * (torch.cuda.device_count() \
                                                                            if cfg.training.use_cuda else 1),  
                              num_workers=cfg.training.num_workers,
                              pin_memory=cfg.training.use_cuda,
                              shuffle=False
                              )
        
        elif category == "classifier":
            dataloader = {
                "rnn": ClassifierStockDatasetRNN,
                "cnn": ClassifierStockDatasetCNN,
                "ffnn": ClassifierStockDatasetFFNN
            }
            return DataLoader(dataloader[model_class](X, y),
                              batch_size=cfg.training.batch_size * (torch.cuda.device_count() \
                                                                            if cfg.training.use_cuda else 1),  
                              num_workers=cfg.training.num_workers,
                              pin_memory=cfg.training.use_cuda,
                              shuffle=False
                              )