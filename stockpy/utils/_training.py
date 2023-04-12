import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from typing import Union, Tuple

import pyro
from pyro.infer import (
    SVI,
    Trace_ELBO
)
from pyro.optim import ClippedAdam
from tqdm.auto import tqdm
from ..config import shared, training
from ._model import Model

class Trainer(Model):
    def __init__(self, 
                 model = None,
                 **kwargs
                 ):
        
        super().__init__(model, **kwargs)
        self._optimizer = self._initOptimizer()
        pyro.clear_param_store()
        self._optimizer = self._initOptimizer()
        if self.type == "probabilistic":
            self._svi = self._initSVI()
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
        if self._model.model_type == "probabilistic":
            adam_params = {"lr": shared.lr, 
                            "betas": shared.betas,
                            "lrd": shared.lrd,
                            "weight_decay": shared.weight_decay
                        }
            return ClippedAdam(adam_params)
        elif self._model.model_type == "neural_network":
            return torch.optim.Adam(self._model.parameters(), 
                                    lr=shared.lr, 
                                    betas=shared.betas, 
                                    eps=shared.eps, 
                                    weight_decay=shared.weight_decay, 
                                    amsgrad=False)
        else:
            raise ValueError("Model type not recognized")
        
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
        if self.name == 'BayesianNN':
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

        This method initializes a learning rate scheduler depending on the model type. For probabilistic models, an instance of
        pyro.optim.ExponentialLR is created, which reduces the learning rate by a factor of gamma for each epoch. For neural network
        models, an instance of torch.optim.lr_scheduler.StepLR is created, which reduces the learning rate by a factor of gamma after 
        every specified step size.

        Raises:
            ValueError: If the model type is not recognized

        Returns:
            Union[pyro.optim.ExponentialLR, torch.optim.lr_scheduler.StepLR]: The learning rate scheduler used to control the learning rate during training.
        """
        if self.type == "probabilistic":
            return pyro.optim.ExponentialLR({'optimizer': self._optimizer, 
                                            'optim_args': {'lr': shared.optim_args}, 
                                            'gamma': shared.gamma}
                                            )
        elif self.type == "neural_network":
            return torch.optim.lr_scheduler.StepLR(self._optimizer, 
                                                step_size=shared.step_size, 
                                                gamma=shared.gamma
                                                )
        else:
            raise ValueError("Model type not recognized")
        
    def _train(self,
               train_dl : torch.utils.data.DataLoader,
               val_dl : torch.utils.data.DataLoader) -> None:

        if self.name!= 'GaussianHMM': 
            self._model.train()

        if self.name == 'DeepMarkovModel':
            self._model.rnn.train()

        best_loss = float('inf')
        counter = 0

        for epoch_ndx in tqdm((range(1, training.epochs + 1)), position=0, leave=True):
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

            if epoch_ndx % training.validation_cadence != 0:                
                tqdm.write(f"Epoch {epoch_ndx}, Loss: {epoch_loss / len(train_dl)}", 
                           end='\r')

            else:
                val_loss = self._doValidation(val_dl)

                tqdm.write(f"Epoch {epoch_ndx}, Val Loss {val_loss}",
                           end='\r')

                # Early stopping
                stop, best_loss, counter = self._earlyStopping(val_loss, 
                                                               best_loss, 
                                                               counter
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
        if self.name != 'GaussianHMM': 
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
            self._saveModel(self.name)
            counter = 0
        else:
            counter += 1

        if counter >= training.patience:
            print(f"No improvement after {training.patience} epochs. Stopping early.")
            return True, best_loss, counter
        else:
            return False, best_loss, counter