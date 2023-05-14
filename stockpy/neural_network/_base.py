import os
import glob
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Union, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from abc import ABCMeta, abstractmethod
from ..base import BaseEstimator
from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..config import Config as cfg

class BaseNN(BaseEstimator, nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        BaseEstimator.__init__(self, **kwargs)

    def _initComponent(self):
        """
        Initializes the model, optimizer, and scheduler.
        """
        self.optimizer = self._initOptimizer()
        self.scheduler = self._initScheduler()

    def _initOptimizer(self) -> torch.optim.Optimizer:
        """
        Initializes the optimizer used to train the model.
        This method initializes the optimizer based on the type of the model (probabilistic or neural network). 
        For probabilistic models, a ClippedAdam optimizer is used with the specified learning rate, betas, learning rate decay, and weight decay.
        For neural network models, the Adam optimizer is used with the specified learning rate, betas, epsilon, and weight decay.
        Returns:
            torch.optim.Optimizer: The optimizer instance used to train the model.
        Raises:
            ValueError: If the model type is not recognized.
        """
        return torch.optim.Adam(self.parameters(), 
                                lr=cfg.training.lr, 
                                betas=cfg.training.betas, 
                                eps=cfg.training.eps, 
                                weight_decay=cfg.training.weight_decay, 
                                amsgrad=False
                                )

    def _initScheduler(self) -> torch.optim.lr_scheduler.StepLR:
        """
        Initializes a learning rate scheduler to control the learning rate during training.
        This method initializes a learning rate scheduler depending on the model type. For probabilistic models, an instance of
        pyro.optim.ExponentialLR is created, which reduces the learning rate by a factor of gamma for each epoch. For neural network
        models, an instance of torch.optim.lr_scheduler.StepLR is created, which reduces the learning rate by a factor of gamma after 
        every specified step size.
        Raises:
            ValueError: If the model type is not recognized
        Returns:
            Union[pyro.optim.ExponentialLR, torch.optim.lr_scheduler.StepLR]: The learning rate scheduler used to control the learning rate during training.
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min') 

    @abstractmethod
    def _initLoss(self):
        pass

    def _log_build_file_path(self):
        file_path_configs = {
            "file_format": self.name + '_{}_{}_{}_{}_{}_{}_{}.state',
            "args": (self.input_size, cfg.nn.hidden_size, self.output_size,
                        cfg.nn.num_layers, cfg.comm.dropout, cfg.training.lr, cfg.training.weight_decay)
        }

        return file_path_configs
    
    def _log_model_state(self):
        state = {
            'model_state': self.state_dict(),
            'model_name': type(self).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
        }

        return state


class ClassifierNN(BaseNN, ClassifierMixin, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = self._initLoss()

    def _initLoss(self):
        return torch.nn.CrossEntropyLoss()

    def _doTraining(self,
                    train_dl: torch.utils.data.DataLoader) -> float:
        """
        Trains the model on the training data for the specified number of epochs.
        Args:
            train_dl (torch.utils.data.DataLoader): The training data.
        Returns:
            float: The training loss.
        """
        train_loss = 0.0
        correct = 0
        total = 0
        true_labels = []
        pred_labels = []

        self.train()
        for x_batch, y_batch in train_dl:
            self.optimizer.zero_grad()
            y_pred = self.forward(x_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(y_pred.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            true_labels.extend(y_batch.tolist())
            pred_labels.extend(predicted.tolist())

        train_loss /= len(train_dl)
        train_f1 = f1_score(true_labels, pred_labels, average='weighted') * 100

        return train_loss, train_f1, true_labels, pred_labels

    def _doValidation(self,
                    val_dl: torch.utils.data.DataLoader):
        """
        Validates the model on the validation data.
        Args:
            val_dl (torch.utils.data.DataLoader): The validation data.
        Returns:
            float: The validation loss.
            float: The validation F1-score.
        """
        val_loss = 0.0
        correct = 0
        total = 0
        true_labels = []
        pred_labels = []
        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                y_pred = self.forward(x_batch)
                loss = self.criterion(y_pred, y_batch)
                val_loss += loss.item()

                _, predicted = torch.max(y_pred.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                true_labels.extend(y_batch.tolist())
                pred_labels.extend(predicted.tolist())
        
        val_loss /= len(val_dl)
        val_f1 = f1_score(true_labels, pred_labels, average='weighted') * 100

        return val_loss, val_f1, true_labels, pred_labels

class RegressorNN(BaseNN, RegressorMixin, metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = self._initLoss()

    def _initLoss(self):
        return torch.nn.MSELoss()
    
    def _doTraining(self,
                    train_dl: torch.utils.data.DataLoader) -> float:
        """
        Trains the model using the given training DataLoader.

        This method trains the model using the given training DataLoader. The method sets the model to training mode, iterates over the input data in batches,
        and performs a forward pass and backward pass for each batch. The method returns the average training loss over all batches.

        Parameters:
            train_dl (torch.utils.data.DataLoader): The training DataLoader containing the input data to train the model on.

        Returns:
            float: The average training loss over all batches.
        """
        train_loss = 0.0
        self.train()
        for x_batch, y_batch in train_dl:
            self.optimizer.zero_grad()
            y_pred = self.forward(x_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_dl)
        return train_loss, None, None, None

    def _doValidation(self,
                      val_dl: torch.utils.data.DataLoader) -> float:
        """
        Validates the model using the given validation DataLoader.

        This method validates the model using the given validation DataLoader. The method sets the model to evaluation mode, iterates over the input data in batches,
        and performs a forward pass for each batch. The method returns the average validation loss over all batches.

        Parameters:
            val_dl (torch.utils.data.DataLoader): The validation DataLoader containing the input data to validate the model on. 
        
        Returns:
            float: The average validation loss over all batches.
        """
        val_loss = 0.0
        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                y_pred = self.forward(x_batch)
                loss = self.criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_dl)
        return val_loss, None, None, None
    
    def _predict(self,
                 test_dl : torch.utils.data.DataLoader
                 ) -> torch.Tensor:
        
        output = torch.tensor([]).to(cfg.training.device)
        self.eval()

        with torch.no_grad():
            for x_batch in test_dl:
                x_batch = x_batch.to(cfg.training.device)
                y_star = self.forward(x_batch)
                output = torch.cat((output, y_star), 0)

        return output
    
