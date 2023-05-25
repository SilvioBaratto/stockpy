import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Union, Tuple, List, Optional
import torch
from enum import Enum
from ._dataset import StockDatasetRNN
from ._dataset import StockDatasetFFNN
from ._dataset import StockDatasetCNN
from ._scaler import ZScoreNormalizer
from ._scaler import MinMaxNormalizer
from ._scaler import RobustScaler
from ..config import Config as cfg

class StockScaler:
    """
    Stock Scaler.
    
    This class is responsible for normalizing and denormalizing the features and targets of stock data. 
    It supports Z-score, Min-Max, and Robust scaling.

    Attributes:
        scaler_type (str): The type of scaler to use. It is set during initialization.
        X_normalizer (TransformMixin): The normalizer for the features. It is set during initialization.
        y_normalizer (TransformMixin): The normalizer for the targets. It is set during initialization.
    """

    def __init__(self): 
        """
        Initializes the StockScaler instance.

        The type of scaler is read from the configuration. The feature and target normalizers are initialized 
        based on this type.
        """

        self.scaler_type = cfg.training.scaler_type

        scaler_classes = {
            'zscore': ZScoreNormalizer,
            'minmax': MinMaxNormalizer,
            'robust': RobustScaler,
        }

        if self.scaler_type not in scaler_classes:
            raise ValueError(f'Invalid scaler type: {self.scaler_type}')
        
        self.X_normalizer = scaler_classes[self.scaler_type]()
        self.y_normalizer = scaler_classes[self.scaler_type]()

    def fit_transform(self, 
                      X_train: Union[np.ndarray, pd.core.frame.DataFrame],
                      y_train: Union[np.ndarray, pd.core.frame.DataFrame] = None,
                      task: str = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fits the normalizers to the training data and then transforms the data.

        The task parameter determines the type of task (regression or classification) and adjusts the normalization 
        process accordingly.

        Args:
            X_train (Union[np.ndarray, pd.core.frame.DataFrame]): The training features to fit the normalizer to and then transform.
            y_train (Union[np.ndarray, pd.core.frame.DataFrame], optional): The training targets to fit the normalizer to and then transform. Default is None.
            task (str, optional): The type of task ('regression' or 'classification'). Default is None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The transformed training features and targets. The targets are None if y_train is None.

        Raises:
            RuntimeError: If the fit method has not been called before the transform method.
        """

        X_train = torch.tensor(X_train.values).float()
        self.X_normalizer.fit(X_train)
        X_train = self.X_normalizer.transform(X_train)

        if y_train is not None and task == 'regression':
            y_train_shape = y_train.shape[1] if len(y_train.shape) > 1 and y_train.shape[1] != 1 else 1
            y_train = torch.tensor(y_train.values).reshape(-1, y_train_shape).float()
            self.y_normalizer.fit(y_train)
            y_train = self.y_normalizer.transform(y_train)

        elif task == 'classification' and y_train is not None:
            y_train = torch.tensor(y_train.values).squeeze().long() - 1
            # if y_train.min() != 0:
            #    y_train -= 1
        else:
            y_train = None

        return X_train, y_train

    def transform(self, 
                  X_test: torch.Tensor):
        """
        Applies the feature normalizer to the test data.

        Args:
            X_test (torch.Tensor): The test features to apply the normalizer to.

        Returns:
            torch.Tensor: The transformed test features.

        Raises:
            RuntimeError: If the fit method has not been called before this method.
        """

        return self.X_normalizer.transform(X_test)
        
    def inverse_transform(self,
                          y_pred: torch.Tensor):
        """
        Applies the inverse target normalizer to the predictions.

        This method transforms the predictions by multiplying by the standard deviation and adding the mean. 
        The fit and transform methods must be called before this method.

        Args:
            y_pred (torch.Tensor): The predictions to apply the inverse normalizer to.

        Returns:
            torch.Tensor: The rescaled predictions.

        Raises:
            RuntimeError: If the fit method has not been called before this method.
        """

        return self.y_normalizer.inverse_transform(y_pred)

class StockDataloader:
    """
    Stock DataLoader.

    This class is responsible for managing the loading of stock data for machine learning models. It supports
    different types of models (RNN, FFNN, CNN) and tasks (regression, classification). The data is loaded in
    batches, and the training data can be split into a training set and a validation set.

    Attributes:
        X_train (torch.Tensor): The normalized training features.
        y_train (torch.Tensor): The normalized training targets.
        model_type (str): The type of model (RNN, FFNN, CNN). It is set during initialization.
        task (str): The type of task (regression or classification). It is set during initialization.
    """

    def __init__(self,
                 X: Union[np.ndarray, pd.core.frame.DataFrame],
                 y: Union[np.ndarray, pd.core.frame.DataFrame] = None,
                 model_type: str = None,
                 task: str = None
                 ):
        """
        Initializes the StockDataloader instance.

        The data is fit-transformed using the scaler. The Dataset instance is created based on the model type.

        Args:
            X (Union[np.ndarray, pd.core.frame.DataFrame]): The features of the data.
            y (Union[np.ndarray, pd.core.frame.DataFrame], optional): The targets of the data. Default is None.
            model_type (str, optional): The type of model (RNN, FFNN, CNN). Default is None.
            task (str, optional): The type of task (regression or classification). Default is None.
        """

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
        
        """
        Returns the DataLoader for a given mode.

        For the training and validation modes, the data is split based on the validation size. For the testing mode,
        the features and targets are transformed and then loaded.

        Args:
            X (Union[np.ndarray, pd.core.frame.DataFrame], optional): The features of the test data. Default is None.
            y (Union[np.ndarray, pd.core.frame.DataFrame], optional): The targets of the test data. Default is None.
            mode (str, optional): The mode ('train', 'val', 'test'). Default is 'train'.

        Returns:
            DataLoader: The DataLoader instance for the given mode.

        Raises:
            ValueError: If the mode or task is invalid.
        """

        if mode == 'train':
            start_idx = 0
            end_idx = int((1 - cfg.training.val_size) * len(self.dataset))
            subset = torch.utils.data.Subset(self.dataset, range(start_idx, end_idx))

        elif mode == 'val':
            start_idx = int((1 - cfg.training.val_size) * len(self.dataset))
            end_idx = len(self.dataset)
            subset = torch.utils.data.Subset(self.dataset, range(start_idx, end_idx))

        elif mode == 'test':
            X = torch.tensor(X.values).float()
            X_test = self.scaler.transform(X)
            if self.task == 'regression':
                y = None
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
                raise ValueError(f"Invalid task: {self.task}. Accepted tasks: 'regression', 'classification'.")
            
        else:
            raise ValueError(f"Invalid mode: {mode}. Accepted modes: 'train', 'val', 'test'.")

    
        return DataLoader(subset,
                          batch_size=cfg.training.batch_size,
                          shuffle=cfg.training.shuffle
                          )


    def inverse_transform_output(self, y_pred):
        """
        Applies the inverse transform to the output.

        This method is used to denormalize the output of the model.

        Args:
            y_pred (torch.Tensor): The output of the model.

        Returns:
            torch.Tensor: The denormalized output.
        """

        return self.scaler.inverse_transform(y_pred)