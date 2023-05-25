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
    """
    This is an abstract base class for all probabilistic models. It extends both the BaseEstimator and PyroModule classes to provide 
    the basic functionalities required by any probabilistic model.

    Methods:
        _initComponent():
            Initializes the model, optimizer, and scheduler.
        _initOptimizer() -> torch.optim.Optimizer:
            Initializes the optimizer used to train the model.
        _initScheduler() -> torch.optim.lr_scheduler.StepLR:
            Initializes a learning rate scheduler to control the learning rate during training.
        _initSVI() -> pyro.infer.svi.SVI:
            Initializes a Stochastic Variational Inference (SVI) instance to optimize the model and guide.
        _log_build_file_path():
            Constructs the configuration for the file path for logging.
        _log_model_state():
            Retrieves the state of the model for logging purposes.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the probabilistic model.
        This method is abstract and must be overridden in subclasses.

        Parameters:
            kwargs (dict): A dictionary of keyword arguments.
        """

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

        Returns:
            torch.optim.Optimizer: The optimizer instance used to train the model.
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

        Returns:
            torch.optim.lr_scheduler.StepLR: The learning rate scheduler used to control the learning rate during training.
        """

        step_lr = StepLR(self.optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma)
        scheduler = PyroLRScheduler(step_lr, self.optimizer)
        return scheduler
    
    @abstractmethod
    def _initSVI(self) -> pyro.infer.svi.SVI:
        """
        Initializes a Stochastic Variational Inference (SVI) instance to optimize the model and guide.

        Returns:
            pyro.infer.svi.SVI: The SVI instance used to optimize the model and guide.
        """

        pass

    def _log_build_file_path(self):
        """
        Constructs the configuration for the file path for logging.

        Returns:
            dict: The configuration dictionary for the file path for logging.
        """

        file_path_configs = {
            "file_format": self.name + '_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.state',
            "args": (self.input_size, cfg.prob.hidden_size, self.output_size,
                    cfg.prob.rnn_dim, cfg.prob.z_dim, cfg.prob.emission_dim,
                    cfg.prob.transition_dim, cfg.prob.variance, cfg.comm.dropout, 
                    cfg.training.lr, cfg.training.weight_decay)
        }

        return file_path_configs
    
    def _log_model_state(self):
        """
        Retrieves the state of the model for logging purposes.

        Returns:
            dict: The dictionary containing the state of the model for logging.
        """

        state = {
            'model_state': self.state_dict(),
            'model_name': type(self).__name__,
            'optimizer_state': self.optimizer.get_state(),
            'optimizer_name': type(self.optimizer).__name__,
        }

        return state
    
class ClassifierProb(BaseProb, ClassifierMixin, metaclass=ABCMeta):
    """
    This is an abstract class for a probabilistic classifier model. It inherits from the `BaseProb` class, 
    the `ClassifierMixin` class, and uses the `ABCMeta` metaclass.

    Methods:
        __init__(self, **kwargs):
            Initializes the classifier by calling the super() function to inherit methods and properties 
            from the parent classes.
    
        _initSVI(self):
            This is an abstract method and needs to be implemented in any child class. This method is 
            responsible for initializing the Stochastic Variational Inference (SVI) instance used for 
            optimization.

        _doTraining(self, train_dl: torch.utils.data.DataLoader) -> float:
            Trains the model on the training data for the specified number of epochs. The method computes 
            the total training loss, and the F1-score using the true and predicted labels. The model is 
            set to training mode and the SVI optimizer is used for training.

        _doValidation(self, val_dl: torch.utils.data.DataLoader) -> float:
            Validates the model on the validation data. The method computes the total validation loss, and 
            the F1-score using the true and predicted labels. The model is set to evaluation mode and the 
            SVI optimizer is used for validation.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the classifier by calling the super() function to inherit methods and properties 
        from the parent classes.

        Parameters:
            kwargs (dict): A dictionary of keyword arguments.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def _initSVI(self):
        """
        This is an abstract method and needs to be implemented in any child class. This method is 
        responsible for initializing the Stochastic Variational Inference (SVI) instance used for 
        optimization.
        """
        pass
    
    def _doTraining(self, train_dl: torch.utils.data.DataLoader) -> float:
        """
        Trains the model on the training data for the specified number of epochs.

        Parameters:
            train_dl (torch.utils.data.DataLoader): The training data.

        Returns:
            float: The training loss.
            float: The F1-score of the training data.
            list: The true labels of the training data.
            list: The predicted labels of the training data.
        """
        # Initialize variables for tracking loss, correct predictions, total samples, and labels
        train_loss = 0.0
        correct = 0
        total = 0
        true_labels = []
        pred_labels = []

        # Switch the model to training mode. This has any effect only on certain modules like Dropout or BatchNorm.
        self.train()

        # Iterate over the training data loader
        for x_batch, y_batch in train_dl:
            # The Stochastic Variational Inference (SVI) optimizer takes a step using the data and labels, and the loss is computed
            loss = self.svi.step(x_batch, y_batch)
            # Add the loss to the total training loss
            train_loss += loss
         
            # Make predictions using the current batch of data
            output = self.forward(x_batch)
            # Get the predicted class by finding the maximum value of the output
            _, predicted = torch.max(output.data, 1)
            # Add the number of data points in the batch to the total number of data points
            total += y_batch.size(0)
            # Add the number of correct predictions in the batch to the total number of correct predictions
            correct += (predicted == y_batch).sum().item()

            # Extend the list of true labels and predicted labels
            true_labels.extend(y_batch.tolist())
            pred_labels.extend(predicted.tolist())

        # Compute the average training loss
        train_loss /= len(train_dl)
        # Compute the weighted F1-score, a measure of the model's performance
        train_f1 = f1_score(true_labels, pred_labels, average='weighted') * 100

        # Return the training loss, F1-score, true labels, and predicted labels
        return train_loss, train_f1, true_labels, pred_labels

    def _doValidation(self, val_dl: torch.utils.data.DataLoader) -> float:
        """
        Validates the model on the validation data.

        Parameters:
            val_dl (torch.utils.data.DataLoader): The validation data.

        Returns:
            float: The validation loss.
            float: The F1-score of the validation data.
            list: The true labels of the validation data.
            list: The predicted labels of the validation data.
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
                # Compute the loss for the batch
                loss = self.svi.evaluate_loss(x_batch, y_batch)
                val_loss += loss

                # Forward pass to obtain model predictions
                output = self.forward(x_batch)
                # Get the predicted labels by selecting the maximum value along the second dimension
                _, predicted = torch.max(output.data, 1)
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

class RegressorProb(BaseProb, RegressorMixin, metaclass=ABCMeta):
    """
    This is an abstract class for a probabilistic regression model. It inherits from the `BaseProb` and 
    `RegressorMixin` classes, and uses the `ABCMeta` metaclass.

    Methods:
        __init__(self, **kwargs):
            Initializes the regressor by calling the super() function to inherit methods and properties 
            from the parent classes.
    
        _initSVI(self):
            This is an abstract method and needs to be implemented in any child class. This method is 
            responsible for initializing the Stochastic Variational Inference (SVI) instance used for 
            optimization.

        _doTraining(self, train_dl: torch.utils.data.DataLoader) -> float:
            Trains the model on the training data. The method computes the total training loss. The model 
            is set to training mode and the SVI optimizer is used for training.

        _doValidation(self, val_dl: torch.utils.data.DataLoader) -> float:
            Validates the model on the validation data. The method computes the total validation loss. 
            The model is set to evaluation mode and the SVI optimizer is used for validation.
        
        _predictNN(self, test_dl: torch.utils.data.DataLoader) -> torch.Tensor:
            Makes a prediction using a neural network model. The data is passed through the model and the 
            output is returned.
        
        _predictHMM(self, test_dl: torch.utils.data.DataLoader) -> torch.Tensor:
            Makes a prediction using a Hidden Markov Model. The data is passed through the model and the 
            output is returned.

        _predict(self, test_dl: torch.utils.data.DataLoader) -> torch.Tensor:
            This is an abstract method and needs to be implemented in any child class. This method is 
            responsible for making predictions.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the regressor by calling the super() function to inherit methods and properties 
        from the parent classes.

        Parameters:
            kwargs (dict): A dictionary of keyword arguments.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def _initSVI(self):
        """
        This is an abstract method and needs to be implemented in any child class. This method is 
        responsible for initializing the Stochastic Variational Inference (SVI) instance used for 
        optimization.
        """
        pass
    
    def _doTraining(self, train_dl: torch.utils.data.DataLoader) -> float:
        """
        Trains the model on the training data for the specified number of epochs.

        Parameters:
            train_dl (torch.utils.data.DataLoader): The training data.

        Returns:
            float: The training loss.
            None: Additional metrics (not used in regression).
            None: Additional metrics (not used in regression).
            None: Additional metrics (not used in regression).
        """
        # Initialize the variable for tracking the training loss
        train_loss = 0.0
        # Set the model to training mode (enables gradient computation and dropout)
        self.train()
        # Iterate over the training data loader
        for x_batch, y_batch in train_dl:
            # Move the batch to the appropriate device
            x_batch = x_batch.to(cfg.training.device)
            y_batch = y_batch.to(cfg.training.device)

            # Compute the loss and perform a single optimization step
            loss = self.svi.step(x_batch, y_batch)
            # Accumulate the training loss
            train_loss += loss

        # Compute the average training loss
        train_loss /= len(train_dl)
        # Return the training loss and None values for additional metrics
        return train_loss, None, None, None

    def _doValidation(self, val_dl: torch.utils.data.DataLoader) -> float:
        """
        Validates the model on the validation data.

        Parameters:
            val_dl (torch.utils.data.DataLoader): The validation data.

        Returns:
            float: The validation loss.
            None: Additional metrics (not used in regression).
            None: Additional metrics (not used in regression).
            None: Additional metrics (not used in regression).
        """
        # Initialize the variable for tracking the validation loss
        val_loss = 0.0
        # Set the model to evaluation mode (disables gradient computation and dropout)
        self.eval()
        # Disable gradient tracking for efficiency
        with torch.no_grad():
            # Iterate over the validation data loader
            for x_batch, y_batch in val_dl:
                # Move the batch to the appropriate device
                x_batch = x_batch.to(cfg.training.device)
                y_batch = y_batch.to(cfg.training.device)

                # Compute the loss for the batch
                loss = self.svi.evaluate_loss(x_batch, y_batch)
                # Accumulate the validation loss
                val_loss += loss

        # Compute the average validation loss
        val_loss /= len(val_dl)

        # Return the validation loss and None values for additional metrics
        return val_loss, None, None, None
    
    def _predictNN(self, test_dl: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Makes predictions on the test data using the model's forward method and Pyro's Predictive class. 
        This method is suitable for Neural Network models.

        Parameters:
            test_dl (torch.utils.data.DataLoader): DataLoader object that contains the test data in batches.

        Returns:
            torch.Tensor: Tensor containing the model's predictions.
        """
        # Initializes an empty tensor to store the output
        output = torch.tensor([])

        # Iterates over the test data loader
        for x_batch in test_dl:
            # Moves the batch data to the specified device (CPU or GPU)
            x_batch = x_batch.to(cfg.training.device)
            
            # Initializes a Predictive instance using the model's forward method, the guide, 
            # the number of samples (equal to the batch size), and the sites to return
            predictive = Predictive(model=self.forward, 
                                    guide=self.guide, 
                                    num_samples=cfg.training.batch_size,
                                    return_sites=("linear.weight", "obs", "_RETURN")
                                    )
            
            # Makes predictions on the batch data
            samples = predictive(x_batch)
            
            # Initializes a dictionary to store the statistics of the sites
            site_stats = {}
            for k, v in samples.items():
                # Stores the mean of the samples for each site
                site_stats[k] = {"mean": torch.mean(v, 0)}

            # Gets the mean prediction of the '_RETURN' site
            y_pred = site_stats['_RETURN']['mean']
            
            # Concatenates the prediction to the output tensor
            output = torch.cat((output, y_pred), 0)
            
        return output
    
    def _predictHMM(self, test_dl: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Makes predictions on the test data using the model's guide and emitter methods. 
        This method is suitable for Hidden Markov Models.

        Parameters:
            test_dl (torch.utils.data.DataLoader): DataLoader object that contains the test data in batches.

        Returns:
            torch.Tensor: Tensor containing the model's predictions.
        """
        # Creates a list to store the predicted y values
        output = []

        # Iterates over the test data in batches
        for x_batch in test_dl:
            # Makes predictions for the current batch
            with torch.no_grad():
                # Computes the location and scale parameters of the latent variable distribution at each time step
                *_, z_loc, z_scale = self.guide(x_batch)
                
                # Applies the softplus function to the scale parameter to ensure its positivity
                z_scale = F.softplus(z_scale)
                
                # Samples from the normal distribution specified by the location and scale parameters
                z_t = dist.Normal(z_loc, z_scale).rsample()
                
                # Passes the latent variable and the data through the emitter to obtain the mean of the emission distribution at each time step
                mean_t, _ = self.emitter(z_t, x_batch)
                    
                # Gets the mean of the emission distribution at the last time step
                mean_last = mean_t[:, -1, :]

            # Adds the predicted y values for the current batch to the list
            output.append(mean_last)

        # Concatenates the predicted y values for all batches into a single tensor
        output = torch.cat(output)

        # Reshapes the tensor to get an array of shape [number_of_samples, 1]
        output = output.reshape(-1, 1)

        # Returns the predicted y values as a tensor
        return output
    
    @abstractmethod
    def _predict(self, test_dl: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Makes predictions on the test data.

        Parameters:
            test_dl (torch.utils.data.DataLoader): The test data.

        Returns:
            torch.Tensor: The predicted output.
        """
        pass