import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score
from typing import Union, Tuple, Callable
from pyro.infer import Predictive
import pyro.distributions as dist
import torch.nn.functional as F

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

class Probabilistic(Model):
    def __init__(self, 
                 model=None,
                 **kwargs
                 ):
        
        super().__init__(model, **kwargs)
        pyro.clear_param_store()
        self._optimizer = self._initOptimizer()
        self._svi = self._initSVI()
        self._scheduler = self._initScheduler()

        if self.type != "probabilistic":
            raise ValueError("Model type not recognized")
        
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
        adam_params = {"lr": cfg.shared.lr, 
                            "betas": cfg.shared.betas,
                            "lrd": cfg.shared.lrd,
                            "weight_decay": cfg.shared.weight_decay
                        }
        return ClippedAdam(adam_params)
        
    def _initSVI(self) -> pyro.infer.svi.SVI:
        """
        Initializes a Stochastic Variational Inference (SVI) instance to optimize the model and guide.

        This method initializes an SVI instance using the model and guide, the optimizer, and the Trace_ELBO loss function. 
        The SVI instance is used to perform stochastic variational inference, which is an optimization-based approach to approximate 
        posterior distributions for Bayesian models.

        If the model is a BayesianNN, the SVI is initialized with the model directly. For other model types, the SVI is initialized 
        with the model's underlying model attribute.

        Returns:
            pyro.infer.svi.SVI: The SVI instance used to optimize the model and guide.
        """
        model_to_use = self._model if self.name == 'BayesianNNRegressor' else self._model.model

        return SVI(model_to_use, 
                self._guide, 
                self._initOptimizer(), 
                loss=Trace_ELBO()
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
        return pyro.optim.ExponentialLR({'optimizer': self._optimizer, 
                                        'optim_args': {'lr': cfg.shared.optim_args}, 
                                        'gamma': cfg.shared.gamma}
                                        )
    
    def _trainRegressor(self,
                        train_dl : torch.utils.data.DataLoader) -> float:
       
        train_loss = 0.0
        if self.name == 'DeepMarkovModel':
            self._model.rnn.train()
        for x_batch, y_batch in train_dl:
            loss = self._computeBatchLossRegressor(x_batch, y_batch)
            train_loss += loss
        
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
        return self._svi.step(
                x_data=x_batch,
                y_data=y_batch
                )
    
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
        if self.name != 'GaussianHMM': 
            self._model.eval()
        with torch.no_grad():  
            for x_batch, y_batch in val_dl:
                loss = self._svi.evaluate_loss(x_batch, y_batch)
                val_loss += loss               

        return val_loss / len(val_dl)
    
    def _earlyStopping(self,
                       total_loss: float,
                       best_loss: float,
                       counter: int
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
            self._saveModel(self._model.name, self._optimizer)
            counter = 0
        else:
            counter += 1

        if counter >= cfg.training.patience:
            print(f"No improvement after {cfg.training.patience} epochs. Stopping early.")
            return True, best_loss, counter
        else:
            return False, best_loss, counter
        
    def _predictRegressor(self,
                          test_dl : torch.utils.data.DataLoader
                          ) -> torch.Tensor:
        """
        Predict target values for the given validation DataLoader using a probabilistic model.

        This method generates predictions using a Bayesian Neural Network or a Deep Markov Model. For the Bayesian Neural
        Network, it uses the Pyro Predictive class to generate samples and compute the mean of the predicted values. For the
        Deep Markov Model, it computes the mean of the emission distribution for each time step and retrieves the mean for
        the last time step.

        Parameters:
            val_dl (torch.utils.data.DataLoader): The validation DataLoader containing the input data to make predictions on.

        Returns:
            torch.Tensor: The predicted target values as a torch.Tensor.
        """     
        if self.name == 'BayesianNNRegressor':
            output = torch.tensor([])
            for x_batch in test_dl:
                x_batch = x_batch.to(cfg.training.device)
                predictive = Predictive(model=self._model, 
                                        guide=self._guide, 
                                        num_samples=cfg.training.batch_size,
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
            for x_batch in test_dl:
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
        
    def _predictClassifier(self,
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
        pass