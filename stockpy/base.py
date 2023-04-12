import torch
import torch.nn as nn
from typing import Union, Tuple
import pandas as pd
import numpy as np

from .config import training
from .utils._dataloader import normalize
from .utils._training import Trainer
from .utils._model import Model
from .utils._predict import Predict
from .utils._dataloader import _initTrainValDl

class Base(Trainer, Predict):

    def __init__(self, 
                 model = None,
                 **kwargs
                 ):
    
        super().__init__(model=model, **kwargs)

    def _modelEval(self):
        print(self._model.eval())
            
    def fit(self, 
            x_train: Union[np.ndarray, pd.core.frame.DataFrame],
            **kwargs
            ) -> None:
        """
        Fits the neural network model to a given dataset.

        This method takes the input training dataset and trains the model for a specified number of epochs. 
        It uses the given batch size and sequence length to preprocess the data for the model. 
        Validation is performed at specified intervals (validation_cadence) 
        to monitor the model's performance on the validation set. 
        Early stopping is implemented using the patience parameter.

        :param x_train: The training dataset, either as a numpy array or pandas DataFrame
        :type x_train: Union[np.ndarray, pd.core.frame.DataFrame]
        :param epochs: The number of epochs to train the model for, defaults to 10
        :type epochs: int, optional
        :param sequence_length: The length of the input sequence, defaults to 30
        :type sequence_length: int, optional
        :param batch_size: The batch size to use during training, defaults to 8
        :type batch_size: int, optional
        :param num_workers: The number of workers to use for data loading, defaults to 4
        :type num_workers: int, optional
        :param validation_sequence: The number of time steps to reserve for validation during training, defaults to 30
        :type validation_sequence: int, optional
        :param validation_cadence: How often to run validation during training, defaults to 5
        :type validation_cadence: int, optional
        :param patience: How many epochs to wait for improvement in validation loss before stopping early, defaults to 5
        :type patience: int, optional

        :return: None
        """
        for key, value in kwargs.items():
            setattr(training, key, value)

        if self.name == "MLP" or self.name == "BayesianNN":
            training.sequence_length = 0
        
        train_dl, val_dl = _initTrainValDl(x_train)

        self._train(train_dl, val_dl)
                        
    def predict(self, 
                x_test: Union[np.ndarray, pd.core.frame.DataFrame]
                ) -> np.ndarray:
        """
        Generate predictions for the given test set using the trained model.

        This public method calls the internal `_predict()` method to generate predictions on the given test set. The test
        set can be provided as a NumPy array or a pandas DataFrame. The returned predictions are in the form of a NumPy
        array.

        Parameters:
            x_test (Union[np.ndarray, pd.core.frame.DataFrame]): The test set to make predictions on, either as a NumPy array or pandas DataFrame.

        Returns:
            np.ndarray: The predicted target values for the given test set, as a NumPy array.
        """
        
        return self._predict(x_test)
