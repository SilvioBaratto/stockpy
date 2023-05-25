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
    """
    This class is an abstract base class for all neural network models.
    It extends both the `BaseEstimator` and `torch.nn.Module` classes to provide the basic functionalities required by any model.

    Methods:
        _initComponent(): Initializes the optimizer and the learning rate scheduler.
        _initOptimizer() -> torch.optim.Optimizer: Initializes the optimizer.
        _initScheduler() -> torch.optim.lr_scheduler.ReduceLROnPlateau: Initializes the learning rate scheduler.
        _initLoss(): Initializes the loss function. This method is abstract and must be overridden in subclasses.
        _log_build_file_path(): Builds the file path for saving the model state.
        _log_model_state(): Logs the state of the model, including the model's parameters and the state of the optimizer.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the neural network model.
        This method is abstract and must be overridden in subclasses.

        Args:
            kwargs (dict): A dictionary of keyword arguments.
        """

        nn.Module.__init__(self)
        BaseEstimator.__init__(self, **kwargs)

    def _initComponent(self):
        """
        Initializes the optimizer and the learning rate scheduler.
        """

        self.optimizer = self._initOptimizer()
        self.scheduler = self._initScheduler()

    def _initOptimizer(self) -> torch.optim.Optimizer:
        """
        Initializes the optimizer.

        Returns:
            optimizer (torch.optim.Optimizer): The initialized optimizer.
        """

        return torch.optim.Adam(self.parameters(), 
                                lr=cfg.training.lr, 
                                betas=cfg.training.betas, 
                                eps=cfg.training.eps, 
                                weight_decay=cfg.training.weight_decay, 
                                amsgrad=False)

    def _initScheduler(self) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """
        Initializes the learning rate scheduler.

        Returns:
            scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): The initialized learning rate scheduler.
        """

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode=cfg.training.scheduler_mode,
            factor=cfg.training.scheduler_factor,
            patience=cfg.training.scheduler_patience,
            threshold=cfg.training.scheduler_threshold
        )

    @abstractmethod
    def _initLoss(self):
        """
        Initializes the loss function.
        This method is abstract and must be overridden in subclasses.
        """

        pass

    def _log_build_file_path(self):
        """
        Builds the file path for saving the model state.

        Returns:
            file_path_configs (dict): A dictionary containing the file format and arguments to be used for building the file path.
        """

        file_path_configs = {
            "file_format": self.name + '_{}_{}_{}_{}_{}_{}_{}.state',
            "args": (self.input_size, cfg.nn.hidden_size, self.output_size,
                        cfg.nn.num_layers, cfg.comm.dropout, cfg.training.lr, cfg.training.weight_decay)
        }

        return file_path_configs
    
    def _log_model_state(self):
        """
        Logs the state of the model, including the model's parameters and the state of the optimizer.

        Returns:
            state (dict): A dictionary containing the state of the model and the optimizer.
        """

        state = {
            'model_state': self.state_dict(),
            'model_name': type(self).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
        }

        return state


class ClassifierNN(BaseNN, ClassifierMixin, metaclass=ABCMeta):
    """
    This class is an abstract base class for all neural network models for classification tasks.
    It extends both the `BaseNN` and `ClassifierMixin` classes to provide the basic functionalities required by any classification model.

    Methods:
        _initLoss(): Initializes the loss function.
        _doTraining(train_dl: torch.utils.data.DataLoader) -> float: Trains the model on the training data.
        _doValidation(val_dl: torch.utils.data.DataLoader): Validates the model on the validation data.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the neural network model for classification.
        This method is abstract and must be overridden in subclasses.

        Args:
            kwargs (dict): A dictionary of keyword arguments.
        """

        super().__init__(**kwargs)
        self.criterion = self._initLoss()

    def _initLoss(self):
        """
        Initializes the loss function.

        Returns:
            loss (torch.nn.CrossEntropyLoss): The initialized loss function.
        """

        return torch.nn.CrossEntropyLoss()

    def _doTraining(self, train_dl: torch.utils.data.DataLoader) -> float:
        """
        Trains the model on the training data for the specified number of epochs.

        Args:
            train_dl (torch.utils.data.DataLoader): The training data.

        Returns:
            train_loss (float): The training loss.
            train_f1 (float): The F1-score of the training data.
            true_labels (list): The true labels of the training data.
            pred_labels (list): The predicted labels of the training data.
        """

        # Initialize variables for tracking loss, correct predictions, total samples, and labels
        train_loss = 0.0
        correct = 0
        total = 0
        true_labels = []
        pred_labels = []

        # Set the model to training mode (enables gradient computation and dropout)
        self.train()

        # Iterate over the training data loader
        for x_batch, y_batch in train_dl:
            # Clear the gradients of the optimizer
            self.optimizer.zero_grad()
            # Forward pass to obtain model predictions
            y_pred = self.forward(x_batch)

            # Compute the loss between the predictions and the ground truth
            loss = self.criterion(y_pred, y_batch)
            # Backpropagation: compute gradients and update model parameters
            loss.backward()
            self.optimizer.step()
            # Accumulate the training loss
            train_loss += loss.item()

            # Get the predicted labels by selecting the maximum value along the second dimension
            _, predicted = torch.max(y_pred.data, 1)
            # Update the count of total samples and correct predictions
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            # Extend the true and predicted labels lists
            true_labels.extend(y_batch.tolist())
            pred_labels.extend(predicted.tolist())

        # Compute the average training loss
        train_loss /= len(train_dl)
        # Calculate the weighted F1 score for the true and predicted labels
        train_f1 = f1_score(true_labels, pred_labels, average='weighted') * 100

        # Return the training loss, F1 score, true labels, and predicted labels
        return train_loss, train_f1, true_labels, pred_labels

    def _doValidation(self, val_dl: torch.utils.data.DataLoader):
        """
        Validates the model on the validation data.

        Args:
            val_dl (torch.utils.data.DataLoader): The validation data.

        Returns:
            val_loss (float): The validation loss.
            val_f1 (float): The F1-score of the validation data.
            true_labels (list): The true labels of the validation data.
            pred_labels (list): The predicted labels of the validation data.
        """

        # Initialize variables for tracking loss, correct predictions, total samples, and labels
        val_loss = 0.0
        correct = 0
        total = 0
        true_labels = []
        pred_labels = []

        # Set the model to evaluation mode (disables gradient computation and dropout)
        self.eval()

        # Disable gradient tracking for efficiency
        with torch.no_grad():
            # Iterate over the validation data loader
            for x_batch, y_batch in val_dl:
                # Forward pass to obtain model predictions
                y_pred = self.forward(x_batch)
                # Compute the loss between the predictions and the ground truth
                loss = self.criterion(y_pred, y_batch)
                val_loss += loss.item()

                # Get the predicted labels by selecting the maximum value along the second dimension
                _, predicted = torch.max(y_pred.data, 1)
                # Update the count of total samples and correct predictions
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                # Extend the true and predicted labels lists
                true_labels.extend(y_batch.tolist())
                pred_labels.extend(predicted.tolist())

        # Compute the average validation loss
        val_loss /= len(val_dl)
        # Calculate the weighted F1 score for the true and predicted labels
        val_f1 = f1_score(true_labels, pred_labels, average='weighted') * 100

        # Return the validation loss, F1 score, true labels, and predicted labels
        return val_loss, val_f1, true_labels, pred_labels

class RegressorNN(BaseNN, RegressorMixin, metaclass=ABCMeta):
    """
    This class is an abstract base class for all neural network models for regression tasks.
    It extends both the `BaseNN` and `RegressorMixin` classes to provide the basic functionalities required by any regression model.

    Methods:
        _initLoss(): Initializes the loss function.
        _doTraining(train_dl: torch.utils.data.DataLoader) -> float: Trains the model on the training data.
        _doValidation(val_dl: torch.utils.data.DataLoader) -> float: Validates the model on the validation data.
        _predict(test_dl: torch.utils.data.DataLoader) -> torch.Tensor: Makes predictions on the test data.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the neural network model for regression.
        This method is abstract and must be overridden in subclasses.

        Args:
            kwargs (dict): A dictionary of keyword arguments.
        """

        super().__init__(**kwargs)
        self.criterion = self._initLoss()

    def _initLoss(self):
        """
        Initializes the loss function.

        Returns:
            loss (torch.nn.MSELoss): The initialized loss function.
        """

        return torch.nn.MSELoss()

    def _doTraining(self, train_dl: torch.utils.data.DataLoader) -> float:
        """
        Trains the model on the training data for the specified number of epochs.

        Args:
            train_dl (torch.utils.data.DataLoader): The training data.

        Returns:
            train_loss (float): The training loss.
        """

        # Initialize the variable for tracking the training loss
        train_loss = 0.0
        # Set the model to training mode (enables gradient computation and dropout)
        self.train()

        # Iterate over the training data loader
        for x_batch, y_batch in train_dl:
            # Clear the gradients of the optimizer
            self.optimizer.zero_grad()
            # Forward pass to obtain model predictions
            y_pred = self.forward(x_batch)

            # Compute the loss between the predictions and the ground truth
            loss = self.criterion(y_pred, y_batch)
            # Backpropagation: compute gradients and update model parameters
            loss.backward()
            self.optimizer.step()
            # Accumulate the training loss
            train_loss += loss.item()

        # Compute the average training loss
        train_loss /= len(train_dl)
        # Return the training loss and None values for additional metrics
        return train_loss, None, None, None

    def _doValidation(self, val_dl: torch.utils.data.DataLoader) -> float:
        """
        Validates the model on the validation data.

        Args:
            val_dl (torch.utils.data.DataLoader): The validation data.

        Returns:
            val_loss (float): The validation loss.
        """

        # Initialize the variable for tracking the validation loss
        val_loss = 0.0
        # Set the model to evaluation mode (disables gradient computation and dropout)
        self.eval()
        # Disable gradient tracking for efficiency
        with torch.no_grad():
            # Iterate over the validation data loader
            for x_batch, y_batch in val_dl:
                # Forward pass to obtain model predictions
                y_pred = self.forward(x_batch)
                # Compute the loss between the predictions and the ground truth
                loss = self.criterion(y_pred, y_batch)
                # Accumulate the validation loss
                val_loss += loss.item()

        # Compute the average validation loss
        val_loss /= len(val_dl)
        # Return the validation loss and None values for additional metrics
        return val_loss, None, None, None

    def _predict(self, test_dl: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Makes predictions on the test data.

        Args:
            test_dl (torch.utils.data.DataLoader): The test data.

        Returns:
            output (torch.Tensor): The predicted output.
        """

        # Initialize an empty tensor to store the predicted output
        output = torch.tensor([]).to(cfg.training.device)
        # Set the model to evaluation mode (disables gradient computation and dropout)
        self.eval()
        # Disable gradient tracking for efficiency
        with torch.no_grad():
            # Iterate over the test data loader
            for x_batch in test_dl:
                # Move the batch to the appropriate device
                x_batch = x_batch.to(cfg.training.device)
                # Forward pass to obtain model predictions
                y_star = self.forward(x_batch)
                # Concatenate the predictions to the output tensor
                output = torch.cat((output, y_star), 0)

        # Return the tensor containing the predicted output
        return output
    
