import datetime
import hashlib
import os
import shutil
import sys
import glob
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from typing import Union, Tuple

import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import (
    SVI,
    Trace_ELBO,
    TraceMeanField_ELBO,
    Predictive
)
from pyro.optim import ClippedAdam
from pyro.nn import PyroModule
import torch.nn.functional as F
from torchviz import make_dot

from .utils import StockDataset, normalize
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange
from .config import ModelArgs as args

class ModelTrainer():

    def __init__(self, 
                 model = None,
                 **kwargs
                 ):
        for key, value in kwargs.items():
            setattr(args, key, value)

        self.use_cuda = torch.cuda.is_available()
        self._initModel(model)

    def _modelEval(self):
        print(self._model.eval())
    
    def _initOptimizer(self) -> torch.optim.Optimizer:
        """
        Initializes the optimizer used to train the model.
        Returns:
            optimizer (torch.optim.AdamW): Optimizer instance
        """
        if self.type == "probabilistic":
            adam_params = {"lr": args.lr, 
                            "betas": args.betas,
                            "lrd": args.lrd,
                            "weight_decay": args.weight_decay
                        }
            return ClippedAdam(adam_params)
        elif self.type == "neural_network":
            return torch.optim.Adam(self._model.parameters(), 
                                    lr=args.lr, 
                                    betas=args.betas, 
                                    eps=args.eps, 
                                    weight_decay=args.weight_decay, 
                                    amsgrad=False)
        else:
            raise ValueError("Model type not recognized")
        
    def _initSVI(self) -> pyro.infer.svi.SVI:
        """
        Initializes a Stochastic Variational Inference (SVI) instance to optimize the model and guide.
        Returns:
            svi (pyro.infer.svi.SVI): SVI instance
        """
        if self._model.__class__.__name__[1:] == 'BayesianNN':
            return SVI(self._model, 
                    self._guide, 
                    self._initOptimizer(), 
                    loss=Trace_ELBO()
                    )
        else: 
            return SVI(self._model.model, 
                    self._guide, 
                    self._initOptimizer(), 
                    loss=Trace_ELBO()
                    )
        
    def _initScheduler(self) -> Union[pyro.optim.ExponentialLR, torch.optim.lr_scheduler.StepLR]:
        """
        Initializes a learning rate scheduler to control the learning rate during training.
        Returns:
            scheduler (pyro.optim.ExponentialLR): Learning rate scheduler
        """
        if self.type == "probabilistic":
            return pyro.optim.ExponentialLR({'optimizer': self._optimizer, 
                                            'optim_args': {'lr': args.optim_args}, 
                                            'gamma': args.gamma}
                                            )
        elif self.type == "neural_network":
            return torch.optim.lr_scheduler.StepLR(self._optimizer, 
                                                step_size=args.step_size, 
                                                gamma=args.gamma
                                                )
        else:
            raise ValueError("Model type not recognized")
        
    def _initTrainDl(self, 
                    x_train: Union[np.ndarray, pd.core.frame.DataFrame], 
                    batch_size: int, 
                    num_workers: int, 
                    sequence_length: int
                    ) -> torch.utils.data.DataLoader:
        """
        Initializes the training data loader.
        Parameters:
            x_train (numpy.ndarray or pandas dataset): the training dataset
            batch_size (int): the batch size to use for training
            num_workers (int): the number of workers to use for data loading
            sequence_length (int): the length of the input sequence
        Returns:
            train_dl (torch.utils.data.DataLoader): the training data loader
        """

        train_dl = StockDataset(x_train, sequence_length=sequence_length)

        train_dl = DataLoader(train_dl, 
                            batch_size=batch_size * (torch.cuda.device_count() \
                                                                   if self.use_cuda else 1),  
                            num_workers=num_workers,
                            pin_memory=self.use_cuda,
                            shuffle=True
                            )

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._sequence_length = sequence_length

        return train_dl

    def _initValDl(self, 
                   x_test: Union[np.ndarray, pd.core.frame.DataFrame]
                   )-> torch.utils.data.DataLoader:
        """
        Initializes the validation data loader.
        Parameters:
            x_test (numpy.ndarray or pandas dataset): the validation dataset
        Returns:
            val_dl (torch.utils.data.DataLoader): the validation data loader
        """

        val_dl = StockDataset(x_test, 
                                sequence_length=self._sequence_length
                                )

        val_dl = DataLoader(val_dl, 
                            batch_size=self._batch_size * (torch.cuda.device_count() \
                                                    if self.use_cuda else 1), 
                            num_workers=self._num_workers,
                            pin_memory=self.use_cuda,
                            shuffle=False
                            )
        
        return val_dl
    
    def _initTrainValData(self, 
                          x_train: Union[np.ndarray, pd.core.frame.DataFrame],
                          validation_sequence: int,
                          batch_size: int,
                          num_workers: int,
                          sequence_length: int
                          )-> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Initializes the training and validation data loaders.
        Parameters:
            x_train (numpy.ndarray): the training dataset
            validation_sequence (int): the number of time steps to reserve for validation during training
            batch_size (int): the batch size to use during training
            num_workers (int): the number of workers to use for data loading
            sequence_length (int): the length of the input sequence
        Returns:
            train_dl (torch.utils.data.DataLoader): the training data loader
            val_dl (torch.utils.data.DataLoader): the validation data loader
        """

        scaler = normalize(x_train)

        x_train = scaler.fit_transform()
        val_dl = x_train[-validation_sequence:]
        x_train = x_train[:len(x_train)-len(val_dl)]

        train_dl = self._initTrainDl(x_train, 
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        sequence_length=sequence_length
                                        )

        val_dl = self._initValDl(val_dl)

        return train_dl, val_dl

    def fit(self, 
            x_train: Union[np.ndarray, pd.core.frame.DataFrame],
            epochs : int=10,
            sequence_length : int=30,
            batch_size : int=8, 
            num_workers : int =4,
            validation_sequence : int =30, 
            validation_cadence : int =5,
            patience : int =5
            ) -> None:
        """
        Fits the neural network model to a given dataset.

        Parameters:
            x_train (numpy.ndarray): the training dataset
            epochs (int): the number of epochs to train the model for
            sequence_length (int): the length of the input sequence
            batch_size (int): the batch size to use during training
            num_workers (int): the number of workers to use for data loading
            validation_sequence (int): the number of time steps to reserve for validation during training
            validation_cadence (int): how often to run validation during training
            patience (int): how many epochs to wait for improvement in validation loss before stopping early

        Returns:
            None
        """
        sequence_lengths = {
            "MLP": 0,
            "BayesianNN": 0,
            # Add more model types and sequence lengths here
        }

        sequence_length = sequence_lengths.get(self._model.__class__.__name__[1:], sequence_length)

        train_dl, val_dl = self._initTrainValData(x_train,
                                                validation_sequence,
                                                batch_size,
                                                num_workers,
                                                sequence_length
                                                )

        self._train(epochs,
                    train_dl,
                    val_dl,
                    validation_cadence,
                    patience
                    )
                
    def _train(self,
               epochs : int,
               train_dl : torch.utils.data.DataLoader,
               val_dl : torch.utils.data.DataLoader,
               validation_cadence : int,
               patience : int
               ) -> None:
        
        if self._model.__class__.__name__[1:] != 'GaussianHMM': 
            self._model.train()

        if self._model.__class__.__name__[1:] == 'DeepMarkovModel':
            self._model.rnn.train()

        best_loss = float('inf')
        counter = 0

        for epoch_ndx in tqdm((range(1, epochs + 1)), position=0, leave=True):
            epoch_loss = 0.0
            for x_batch, y_batch in train_dl:   
                if self.type == "neural_network":
                    self._optimizer.zero_grad()  
                    loss = self._computeBatchLoss(x_batch, y_batch)

                    loss.backward()     
                    self._optimizer.step()
                    epoch_loss += loss

                elif self.type == 'probabilistic':
                    loss = self._computeBatchLoss(x_batch, y_batch)
                    epoch_loss += loss

            self._scheduler.step()

            if epoch_ndx % validation_cadence != 0:                
                tqdm.write(f"Epoch {epoch_ndx}, Loss: {epoch_loss / len(train_dl)}", 
                           end='\r')

            else:
                val_loss = self._doValidation(val_dl)

                tqdm.write(f"Epoch {epoch_ndx}, Val Loss {val_loss}",
                           end='\r')

                # Early stopping
                stop, best_loss, counter = self._earlyStopping(val_loss, 
                                                               best_loss, 
                                                               counter, 
                                                               patience,
                                                               epoch_ndx
                                                               )
                if stop:
                    break  

    def _computeBatchLoss(self, 
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
        if self.type == "probabilistic":
            loss = self._svi.step(
                x_data=x_batch,
                y_data=y_batch
            )
        elif self.type == "neural_network":
            output = self._model(x_batch)
            loss_function = nn.MSELoss()
            loss = loss_function(output, y_batch)
        else:
            raise ValueError("Model type not recognized")
        
        return loss
    
    def _doValidation(self, 
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
        if self._model.__class__.__name__[1:] != 'GaussianHMM': 
            self._model.eval()
        with torch.no_grad():  
            for x_batch, y_batch in val_dl:
                if self.type == 'neural_network':
                    output = self._model(x_batch)
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(output, y_batch)
                    val_loss += loss.item()
                elif self.type == 'probabilistic':
                    loss = self._svi.evaluate_loss(x_batch, y_batch)
                    val_loss += loss               

        return val_loss / len(val_dl)

    def _earlyStopping(self,
                       total_loss: float,
                       best_loss: float,
                       counter: int,
                       patience: int,
                       epoch_ndx: int
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
            self._saveModel(self._model.__class__.__name__[1:])
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"No improvement after {patience} epochs. Stopping early.")
            return True, best_loss, counter
        else:
            return False, best_loss, counter
        
    def predict(self, 
                x_test: Union[np.ndarray, pd.core.frame.DataFrame]
                ) -> np.ndarray:
        """
        Make predictions on a given test set.
        Parameters:
            x_test (np.ndarray): the test set to make predictions on
        Returns:
            np.ndarray: the predicted values for the given test set
        """

        scaler = normalize(x_test)
        x_test = scaler.fit_transform()
        val_dl = self._initValDl(x_test)

        if self.type == "probabilistic":
            output = self._predict_probabilistic(val_dl)
        elif self.type == "neural_network":
            output = self._predict_neural_network(val_dl)
        else:
            raise ValueError("Model type not recognized")

        output = output.detach().numpy() * scaler.std() + scaler.mean()
                    
        return output
    
    def _predict_neural_network(self, 
                                val_dl : torch.utils.data.DataLoader
                                ) -> torch.Tensor:
        output = torch.tensor([])
        self._model.eval()
        
        with torch.no_grad():
            for x_batch, _ in val_dl:
                y_star = self._model(x_batch)
                output = torch.cat((output, y_star), 0)
                
        return output
    
    def _predict_probabilistic(self,
                               val_dl : torch.utils.data.DataLoader
                               ) -> torch.Tensor:
        
        if self._model.__class__.__name__[1:] == 'BayesianNN':
            output = torch.tensor([])
            for x_batch, _ in val_dl:
                predictive = Predictive(model=self._model, 
                                        guide=self._guide, 
                                        num_samples=self._batch_size,
                                        return_sites=("linear.weight", 
                                                        "obs", 
                                                        "_RETURN")
                                                    )
                samples = predictive(x_batch)
                site_stats = {}
                for k, v in samples.items():
                    site_stats[k] = {
                        "mean": torch.mean(v, 0)
                    }

                y_pred = site_stats['_RETURN']['mean']
                output = torch.cat((output, y_pred), 0)
            
            return output
        
        else: 
            # create a list to hold the predicted y values
            output = []

            # iterate over the test data in batches
            for x_batch, _ in val_dl:
                # make predictions for the current batch
                with torch.no_grad():
                    # compute the mean of the emission distribution for each time step
                    *_, z_loc, z_scale = self._guide(x_batch)
                    z_scale = F.softplus(z_scale)
                    z_t = dist.Normal(z_loc, z_scale).rsample()
                    mean_t, _ = self._model.emitter(z_t, x_batch)
                    
                    # get the mean for the last time step
                    mean_last = mean_t[:, -1, :]

                # add the predicted y values for the current batch to the list
                output.append(mean_last)

            # concatenate the predicted y values for all batches into a single tensor
            output = torch.cat(output)

            # reshape the tensor to get an array of shape [151,1]
            output = output.reshape(-1, 1)

            # return the predicted y values as a numpy array
            return output

    def _initModel(self, 
                   model : Union[nn.Module, PyroModule]
                   ) -> None:
        """
        Initializes the neural network model.
        Returns:
            None
        """
        
        if args.pretrained:
            path = self._initModelPath(model, model.__class__.__name__[1:])
            model_dict = torch.load(path)
            model.load_state_dict(model_dict['model_state'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            self._model = model.to(device)

        self._model = model

        if self._model.model_type == "neural_network":
            self.type = 'neural_network'
        elif self._model.model_type == "probabilistic":
            self.type = 'probabilistic'
            if self._model.__class__.__name__[1:] == 'BayesianNN':
                self._guide = AutoDiagonalNormal(self._model)
            else:
                self._guide = self._model.guide
        
        pyro.clear_param_store()
        self._optimizer = self._initOptimizer()
        if self.type == "probabilistic":
            self._svi = self._initSVI()
        self._scheduler = self._initScheduler()

    def _saveModel(self, 
                   type_str : str
                   ) -> None:
        """
        Saves the model to disk.
        Parameters:
            type_str (str): a string indicating the type of model
            epoch_ndx (int): the epoch index
        Returns:
            None
        """

        file_path = os.path.join(
            '..', 
            '..', 
            'models', 
            self._model.__class__.__name__[1:], 
            type_str + '_{}_{}.state'.format(args.dropout,
                                                args.weight_decay
                                                ),
            )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self._model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

    def _initModelPath(self, 
                       model : Union[nn.Module, PyroModule], 
                       type_str : str) -> str:
        """
        Initializes the model path.

        Parameters:
            type_str (str): a string indicating the type of model

        Returns:
            str: the path to the initialized model
        """

        model_dir = '../../models/' + model.__class__.__name__[1:]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        local_path = os.path.join(
            '..', 
            '..', 
            'models', 
            model.__class__.__name__[1:], 
            type_str + '_{}_{}.state'.format(args.dropout,
                                                args.weight_decay
                                                ),
            )

        file_list = glob.glob(local_path)
        
        if not file_list:
            raise ValueError(f"No matching model found in {local_path} for the given parameters.")
        
        # Return the most recent matching file
        return file_list[0]
