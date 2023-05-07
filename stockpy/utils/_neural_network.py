import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score
from typing import Union, Tuple, Callable, List
from sklearn.metrics import confusion_matrix

import pyro
from pyro.infer import (
    SVI,
    Trace_ELBO
)
from pyro.optim import ClippedAdam
from tqdm.auto import tqdm
from ._model import Model
from ._dataloader import StockDataset as sd
from ..config import Config as cfg

class NeuralNetwork(Model):
    def __init__(self, 
                 model=None,
                 **kwargs
                 ):

        super().__init__(model, **kwargs)
        self._initModel(model=model)
        self._optimizer = self._initOptimizer()
        self._scheduler = self._initScheduler()
                
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
        return torch.optim.Adam(self._model.parameters(), 
                                lr=cfg.training.lr, 
                                betas=cfg.training.betas, 
                                eps=cfg.training.eps, 
                                weight_decay=cfg.training.weight_decay, 
                                amsgrad=False
                                )
        
    def _initScheduler(self) -> Union[pyro.optim.ExponentialLR, torch.optim.lr_scheduler.StepLR]:
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
        return torch.optim.lr_scheduler.StepLR(self._optimizer, 
                                                step_size=cfg.training.step_size, 
                                                gamma=cfg.training.gamma
                                                )
        
    def _trainRegressor(self,
                        train_dl : torch.utils.data.DataLoader) -> float:
        
        train_loss = 0.0
        self._model.train()
        for x_batch, y_batch in train_dl:
            loss = self._computeBatchLossRegressor(x_batch, y_batch)
            self._optimizer.zero_grad()  
            loss.backward()     
            self._optimizer.step()
            train_loss += loss.item() 

        return train_loss / len(train_dl)

    def _computeBatchLossRegressor(self, 
                         x_batch : torch.Tensor, 
                         y_batch : torch.Tensor
                         ) -> torch.Tensor:
        """
        Computes the loss for a given batch of data.
        Parameters:
            x_batch (torch.Tensor): the input data
            y_batch (torch.Tensor): the target data
        Returns:
            torch.Tensor: the loss for the given batch of data
        """  
        x_batch = x_batch.to(cfg.training.device)
        y_batch = y_batch.to(cfg.training.device)

        output = self._model(x_batch)
        loss_function = nn.MSELoss()
        loss = loss_function(output, y_batch)
        
        return loss
    
    def _doValidationRegressor(self, 
                      val_dl : torch.utils.data.DataLoader
                      ) -> float: 
        """
        Performs validation on a given validation data loader.
        Parameters:
            val_dl (torch.utils.data.DataLoader): the validation data loader
        Returns:
            float: the total loss over the validation set
        """

        val_loss = 0.0
        self._model.eval()
        with torch.no_grad():  
            for x_batch, y_batch in val_dl:
                x_batch = x_batch.to(cfg.training.device)
                y_batch = y_batch.to(cfg.training.device)

                loss = self._computeBatchLossRegressor(x_batch, y_batch)
                val_loss += loss.item()             

        return val_loss / len(val_dl)
    
    def _trainClassifier(self,
                        train_dl: torch.utils.data.DataLoader) -> float:
        
        self._model.train()
        loss_function = nn.CrossEntropyLoss()
        train_loss = 0.0
        train_accuracy = 0.0
        correct_preds = 0
        total_preds = 0
        true_labels = []
        pred_labels = []
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(cfg.training.device)
            y_batch = y_batch.to(cfg.training.device)
            logits = self._model(x_batch)
            loss = loss_function(logits, y_batch)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            train_loss += loss.item()
            # Calculate the number of correct predictions
            _, predicted_labels = torch.max(logits, 1)
            correct_preds += (predicted_labels == y_batch).sum().item()
            total_preds += y_batch.size(0)

            true_labels.extend(y_batch.cpu().numpy())
            pred_labels.extend(predicted_labels.cpu().numpy())

        train_loss /= len(train_dl)
        train_accuracy = correct_preds / total_preds
        train_f1_score = f1_score(true_labels, pred_labels, average='weighted')

        return train_loss, train_accuracy * 100, train_f1_score * 100


    def _computeBatchLossClassifier(self,
                                    x_batch: torch.Tensor,
                                    y_batch: torch.Tensor
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss for a given batch of data.
        Parameters:
            x_batch (torch.Tensor): the input data
            y_batch (torch.Tensor): the target data
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the loss and accuracy for the given batch of data
        """
        loss_function = nn.CrossEntropyLoss()
        x_batch = x_batch.to(cfg.training.device)
        y_batch = y_batch.to(cfg.training.device)

        logits = self._model(x_batch)
        loss = loss_function(logits, y_batch)

        return loss, logits
        
    def _doValidationClassifier(self,
                                val_dl: torch.utils.data.DataLoader
                                ) -> Tuple[float, float, float, List, List]:
        """
        Computes the validation loss and accuracy for the given validation DataLoader.
        Parameters:
            val_dl (torch.utils.data.DataLoader): the DataLoader for the validation set
        Returns:
            Tuple[float, float]: the validation loss and accuracy
        """
        self._model.eval()  # Set the model to evaluation mode
        loss_function = nn.CrossEntropyLoss()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                x_batch = x_batch.to(cfg.training.device)
                y_batch = y_batch.to(cfg.training.device)

                output = self._model(x_batch)
                loss = loss_function(output, y_batch)

                val_loss += loss.item()

                # Calculate the number of correct predictions
                _, predicted_labels = torch.max(output, 1)

                correct_preds += (predicted_labels == y_batch).sum().item()
                total_preds += y_batch.size(0)

                true_labels.extend(y_batch.cpu().numpy())
                pred_labels.extend(predicted_labels.cpu().numpy())

        val_loss /= len(val_dl)
        val_accuracy = correct_preds / total_preds
        val_f1_score = f1_score(true_labels, pred_labels, average='weighted')

        return val_loss, val_accuracy * 100, val_f1_score * 100, true_labels, pred_labels
    
    def _earlyStopping(self,
                       total_loss: float,
                       best_loss: float,
                       counter: int,
                       ) -> Tuple[bool, float, int]:
        """
        Implements early stopping during training.

        Parameters:
            total_loss (float): the total validation loss
            best_loss (float): the best validation loss seen so far
            counter (int): the number of epochs without improvement in validation loss
            patience (int): how many epochs to wait for improvement in validation loss before stopping early
            epoch_ndx (int): the current epoch number

        Returns:
            tuple: a tuple containing a bool indicating whether to stop early, the best loss seen so far, and the current counter value
        """

        if total_loss < best_loss:
            best_loss = total_loss
            self._saveModel(self.name, self._optimizer)
            counter = 0
        else:
            counter += 1

        if counter >= cfg.training.patience:
            print(f"No improvement after {cfg.training.patience} epochs. Stopping early.")
            return True, best_loss, counter
        else:
            return False, best_loss, counter
        
    def _predict(self,
                test_dl : torch.utils.data.DataLoader
                ) -> torch.Tensor:
        """
        Predict target values for the given validation DataLoader using a neural network model.

        This method generates predictions using a neural network model (e.g., BiGRU). The method sets the model to evaluation
        mode, iterates over the input data in batches, and generates predictions for each batch. The predictions are then
        concatenated into a single output tensor.

        Parameters:
            val_dl (torch.utils.data.DataLoader): The validation DataLoader containing the input data to make predictions on.

        Returns:
            torch.Tensor: The predicted target values as a torch.Tensor.
        """
        output = torch.tensor([]).to(cfg.training.device)
        self._model.eval()

        with torch.no_grad():
            for x_batch in test_dl:
                x_batch = x_batch.to(cfg.training.device)
                y_star = self._model(x_batch)
                output = torch.cat((output, y_star), 0)

        return output
    
    def _score(self,
                test_dl : torch.utils.data.DataLoader
                ) -> torch.Tensor:
        """
        Predict target values for the given validation DataLoader using a neural network model.

        This method generates predictions using a neural network model (e.g., BiGRU). The method sets the model to evaluation
        mode, iterates over the input data in batches, and generates predictions for each batch. The predictions are then
        concatenated into a single output tensor.

        Parameters:
            val_dl (torch.utils.data.DataLoader): The validation DataLoader containing the input data to make predictions on.

        Returns:
            torch.Tensor: The predicted target values as a torch.Tensor.
        """
        _, accuracy, f1_score, true_labels, pred_labels = self._doValidationClassifier(test_dl)
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        
        return accuracy, f1_score, conf_matrix