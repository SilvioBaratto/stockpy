import os
import glob
import torch
from pyro.nn import PyroModule
from pyro.infer import Predictive
import pyro.distributions as dist
import torch.nn.functional as F

import pyro
from pyro.nn import PyroModule
from pyro.infer import (
    SVI,
    Trace_ELBO,
    TraceMeanField_ELBO
)
from pyro.optim import ClippedAdam
from pyro.optim import PyroLRScheduler
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from abc import abstractmethod, ABCMeta
from ..base import BaseEstimator
from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..config import Config as cfg

class CombinedMeta(ABCMeta, type(PyroModule)):
    pass

class BaseProb(BaseEstimator, PyroModule, metaclass=CombinedMeta):

    @abstractmethod
    def __init__(self, **kwargs):
        PyroModule.__init__(self)
        BaseEstimator.__init__(self, **kwargs)

    def _initComponent(self):
        """
        Initializes the model, optimizer, and scheduler.
        """
        self.optimizer = self._initOptimizer()
        # self._initScheduler()
        self.svi = self._initSVI()

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
        adam_params = {"lr": cfg.training.lr, 
                            "betas": cfg.training.betas,
                            "lrd": cfg.training.lrd,
                            "weight_decay": cfg.training.weight_decay
                        }
        return ClippedAdam(adam_params)
    
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
        step_lr = StepLR(self.optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma)
        scheduler = PyroLRScheduler(step_lr, self.optimizer)
        return scheduler
    
    @abstractmethod
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
        pass

    def _log_build_file_path(self):
        file_path_configs = {
            "file_format": self.name + '_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.state',
            "args": (self.input_size, cfg.prob.hidden_size, self.output_size,
                    cfg.prob.rnn_dim, cfg.prob.z_dim, cfg.prob.emission_dim,
                    cfg.prob.transition_dim, cfg.prob.variance, cfg.comm.dropout, 
                    cfg.training.lr, cfg.training.weight_decay)
        }

        return file_path_configs
    
    def _log_model_state(self):
        state = {
            'model_state': self.state_dict(),
            'model_name': type(self).__name__,
            'optimizer_state': self.optimizer.get_state(),
            'optimizer_name': type(self.optimizer).__name__,
        }

        return state   
    
class ClassifierProb(BaseProb, ClassifierMixin, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def _initSVI(self):
        pass
    
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
            loss = self.svi.step(x_batch, y_batch)
            train_loss += loss
            
            output = self.forward(x_batch)
            _, predicted = torch.max(output.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            true_labels.extend(y_batch.tolist())
            pred_labels.extend(predicted.tolist())

        train_loss /= len(train_dl)
        train_f1 = f1_score(true_labels, pred_labels, average='weighted') * 100

        return train_loss, train_f1, true_labels, pred_labels

    def _doValidation(self,
                      val_dl: torch.utils.data.DataLoader) -> float:
        """
        Validates the model on the validation data.
        Args:
            val_dl (torch.utils.data.DataLoader): The validation data.
        Returns:
            float: The validation loss.
        """
        val_loss = 0.0
        correct = 0
        total = 0
        true_labels = []
        pred_labels = []

        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                loss = self.svi.evaluate_loss(x_batch, y_batch)
                val_loss += loss

                output = self.forward(x_batch)
                _, predicted = torch.max(output.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                true_labels.extend(y_batch.tolist())
                pred_labels.extend(predicted.tolist())

        val_loss /= len(val_dl)
        val_f1 = f1_score(true_labels, pred_labels, average='weighted') * 100

        return val_loss, val_f1, true_labels, pred_labels

class RegressorProb(BaseProb, RegressorMixin, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def _initSVI(self):
        pass
    
    def _doTraining(self,
                    train_dl: torch.utils.data.DataLoader) -> float:
        train_loss = 0.0
        self.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(cfg.training.device)
            y_batch = y_batch.to(cfg.training.device)
            loss = self.svi.step(x_batch, y_batch)
            train_loss += loss

        train_loss /= len(train_dl)
        return train_loss, None, None, None

    def _doValidation(self,
                      val_dl: torch.utils.data.DataLoader) -> float:
        val_loss = 0.0
        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                x_batch = x_batch.to(cfg.training.device)
                y_batch = y_batch.to(cfg.training.device)
                loss = self.svi.evaluate_loss(x_batch, y_batch)
                val_loss += loss

        val_loss /= len(val_dl)
        return val_loss, None, None, None
    
    def _predictNN(self,
                   test_dl : torch.utils.data.DataLoader
                   ) -> torch.Tensor:

        output = torch.tensor([])
        for x_batch in test_dl:
            x_batch = x_batch.to(cfg.training.device)
            predictive = Predictive(model=self.forward, 
                                    guide=self.guide, 
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
    
    def _predictHMM(self,
                    test_dl : torch.utils.data.DataLoader
                    ) -> torch.Tensor:
        # create a list to hold the predicted y values
        output = []

        # iterate over the test data in batches
        for x_batch in test_dl:
            # make predictions for the current batch
            with torch.no_grad():
                # compute the mean of the emission distribution for each time step
                *_, z_loc, z_scale = self.guide(x_batch)
                z_scale = F.softplus(z_scale)
                z_t = dist.Normal(z_loc, z_scale).rsample()
                mean_t, _ = self.emitter(z_t, x_batch)
                    
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
    
    @abstractmethod
    def _predict(self,
                 test_dl : torch.utils.data.DataLoader
                 ) -> torch.Tensor:
        pass