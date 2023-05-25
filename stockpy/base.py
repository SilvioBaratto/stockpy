import os
import glob
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Union, Tuple
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from .preprocessing._dataloader import StockDataloader
from .config import Config as cfg

class BaseEstimator(metaclass=ABCMeta):
    """
    Base Estimator.

    This class serves as the base class for all estimators. It defines the common interface that all estimators should
    follow. The actual behavior of the methods is defined in the subclasses.

    Attributes:
        None

    Methods:
        __init__(**kwargs):
            Initializes the BaseEstimator instance. It sets the attributes from the keyword arguments to the
            configuration.
        forward(x):
            The forward pass of the model. It is an abstract method.
        _init_model():
            Initializes the model. It is an abstract method.
        _initOptimizer() -> torch.optim.Optimizer:
            Initializes the optimizer. It is an abstract method.
        _initScheduler() -> torch.optim.lr_scheduler.StepLR:
            Initializes the learning rate scheduler. It is an abstract method.
        _initComponent():
            Initializes the components. It is an abstract method.
        _doTraining():
            Performs the training. It is an abstract method.
        _doValidation():
            Performs the validation. It is an abstract method.
        fit(X, y, **kwargs):
            Fits the model to the data. It initializes the dataloader, the model, and then starts the training.
        _train(train_dl, val_dl):
            Trains the model using the training dataloader and validates using the validation dataloader. It performs
            early stopping if enabled.
        _saveModel():
            Saves the model to a file.
        _loadModel():
            Loads the model from a file.
        _earlyStopping(total_loss, best_loss, counter):
            Implements early stopping during training.
        to(device):
            Moves the model to the specified device.
        name:
            Returns the name of the class.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the BaseEstimator instance. It sets the attributes from the keyword arguments to the
        configuration.

        Args:
            **kwargs: Additional keyword arguments to be set as attributes in the Training configuration.

        Returns:
            None
        """
        for key, value in kwargs.items():
            setattr(cfg.nn, key, value)
            setattr(cfg.prob, key, value)

    @abstractmethod
    def forward(self, x):
        """
        The forward pass of the model.

        Args:
            x: The input tensor.

        Returns:
            None
        """
        pass  

    @abstractmethod
    def _init_model(self):
        """
        Initializes the model.

        Returns:
            None
        """
        pass

    @abstractmethod
    def _initOptimizer(self) -> torch.optim.Optimizer:
        """
        Initializes the optimizer.

        Returns:
            torch.optim.Optimizer: The initialized optimizer.
        """
        pass

    @abstractmethod
    def _initScheduler(self) -> torch.optim.lr_scheduler.StepLR:
        """
        Initializes the learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler.StepLR: The initialized learning rate scheduler.
        """
        pass

    @abstractmethod
    def _initComponent(self):
        """
        Initializes the components.

        Returns:
            None
        """
        pass

    @abstractmethod
    def _doTraining(self):
        """
        Performs the training.

        Returns:
            None
        """
        pass

    @abstractmethod
    def _doValidation(self):
        """
        Performs the validation.

        Returns:
            None
        """
        pass

    def fit(self, 
            X: Union[np.ndarray, pd.core.frame.DataFrame],
            y: Union[np.ndarray, pd.core.frame.DataFrame],
            **kwargs
            ) -> None:
        """
        Fit the estimator to the data.

        This method initializes the data loader, the model, and then starts the training process.

        Args:
            X (Union[np.ndarray, pd.core.frame.DataFrame]): The input data.
            y (Union[np.ndarray, pd.core.frame.DataFrame]): The target data.

            Optional keyword arguments:
            eval (bool): Print settings.
            lr (float): Learning rate for the optimizer.
            betas (tuple): Coefficients used for computing running averages of gradient and its square.
            weight_decay (float): Weight decay (L2 penalty).
            eps (float): Term added to the denominator to improve numerical stability.
            amsgrad (bool): Whether to use the AMSGrad variant of the Adam optimizer.
            gamma (float): Multiplicative factor of learning rate decay.
            step_size (float): Period of learning rate decay.
            scheduler_patience (int): The number of epochs to wait for improvement before stopping early.
            min_delta (int): Minimum change in the monitored quantity to qualify as an improvement.
            scheduler (bool): Whether to use a learning rate scheduler.
            scheduler_mode (str): The mode for the learning rate scheduler.
            scheduler_factor (float): The factor for reducing the learning rate.
            scheduler_threshold (float): The threshold for reducing the learning rate.
            lrd (float): Learning rate decay.
            clip_norm (float): Gradient clipping threshold.
            scaler_type (str): The type of scaler to use.
            epochs (int): The number of epochs to train for.
            batch_size (int): The size of the batches for training.
            sequence_length (int): The length of the sequence for training.
            num_workers (int): The number of worker threads to use for data loading.
            validation_cadence (int): The number of epochs between validation checks.
            optim_args (float): Additional arguments for the optimizer.
            shuffle (bool): Whether to shuffle the data before each epoch.
            val_size (float): The size of the validation set.
            early_stopping (bool): Whether to use early stopping.
            pretrained (bool): Whether to load a pre-trained model.
            folder (str): The folder to save the model to.

        Returns:
            None
        """
        for key, value in kwargs.items():
            setattr(cfg.training, key, value)

        self.input_size = X.shape[1]
        self.output_size = len(np.unique(y)) if self.task == "classification" \
                        else (y.shape[1] if y.ndim > 1 else 1)
        
        self.dataloader = StockDataloader(X=X, 
                                          y=y,
                                          model_type=self.model_type,
                                          task=self.task)
        
        train_dl = self.dataloader.get_loader(mode = 'train')
        val_dl = self.dataloader.get_loader(mode = 'val')
        
        self._init_model()

        if cfg.training.eval:
            print(self.eval())

        if cfg.training.pretrained:
            self._loadModel()

        return self._train(train_dl, val_dl)

    def _train(self,
            train_dl: torch.utils.data.DataLoader,
            val_dl: torch.utils.data.DataLoader) -> None:
        """
        Train the model with the given training and validation data loaders.

        Args:
            train_dl (torch.utils.data.DataLoader): The data loader for the training data.
            val_dl (torch.utils.data.DataLoader): The data loader for the validation data.

        Returns:
            None
        """
        best_loss = np.inf
        counter = 0
        self._initComponent()
    
        for epoch_ndx in tqdm((range(1, cfg.training.epochs + 1)), position=0, leave=True):
            train_results = self._doTraining(train_dl)

            if epoch_ndx % cfg.training.validation_cadence != 0:
                self._log_train_progress(epoch_ndx, train_results)
                # self.scheduler.step(train_results[0])  # assumes loss is first result

            else:
                val_results = self._doValidation(val_dl)
                self._log_validation_progress(epoch_ndx, train_results, val_results)
                # self.scheduler.step(val_results[0])  # assumes loss is first result

                if cfg.training.early_stopping:
                    stop, best_loss, counter = self._earlyStopping(val_results[0], 
                                                          best_loss,
                                                          counter)
                    if stop:
                        break

    def _saveModel(self) -> None:
        """
        Save the current state of the model.

        Returns:
            None
        """
        # Define a helper function to build the file path
        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(lib_dir, 'save', file_format.format(*args))
        
        # Determine the library directory based on the `cfg.training.folder` attribute
        if cfg.training.folder is None:
            lib_dir = os.path.dirname(os.path.abspath(__file__))  # directory of the library
        else:
            lib_dir = cfg.training.folder

        # Get the file path configurations from the `_log_build_file_path` method
        file_path_configs = self._log_build_file_path()

        # Build the file path for saving the model
        file_path = build_file_path(file_path_configs["file_format"], *file_path_configs["args"])
                
        # Create the necessary directory structure for the file path
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        # Get the current state of the model
        state = self._log_model_state()

        # Save the model state to the file path
        torch.save(state, file_path)


    def _loadModel(self) -> None:
        """
        Load a previously saved state of the model.

        Returns:
            None
        """
        # Define a helper function to build the file path
        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(lib_dir, 'save', file_format.format(*args))

        # Determine the library directory based on the `cfg.training.folder` attribute
        if cfg.training.folder is None:
            lib_dir = os.path.dirname(os.path.abspath(__file__))  # directory of the library
        else:
            lib_dir = cfg.training.folder

        # Get the file path configurations from the `_log_build_file_path` method
        file_path_configs = self._log_build_file_path()

        # Build the file path for loading the model
        file_path = build_file_path(file_path_configs["file_format"], *file_path_configs["args"])

        # Check if the file exists
        if not os.path.exists(file_path):
            raise ValueError(f"No matching model found in {file_path} for the given parameters.")

        # Load the saved state from the file
        state = torch.load(file_path)

        # Only load the 'model_state' part of the dictionary
        model_state = state['model_state']

        # Load the model state into the current model
        self.load_state_dict(model_state)

    def _earlyStopping(self,
                       total_loss: float,
                       best_loss: float,
                       counter: int,
                       ) -> Tuple[bool, float, int]:
        """
        Implements early stopping.

        Args:
            total_loss (float): The total loss for the current epoch.
            best_loss (float): The best loss seen so far.
            counter (int): The number of epochs without improvement.

        Returns:
            stop (bool): Whether to stop the training early.
            best_loss (float): The updated best loss.
            counter (int): The updated counter.
        """
        if total_loss < best_loss - cfg.training.min_delta:
            best_loss = total_loss
            self._saveModel()
            counter = 0
        else:
            counter += 1

        if counter >= cfg.training.patience:
            print(f"No improvement after {cfg.training.patience} epochs. Stopping early.")
            return True, best_loss, counter
        else:
            return False, best_loss, counter
    
    def to(self, device: torch.device) -> None:
        """
        Moves the model to the specified device.

        Args:
            device (torch.device): The device to move the model to.

        Returns:
            None
        """
        super().to(device)
    
    @property
    def name(self):
        """
        Returns the name of the class.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__

class ClassifierMixin(BaseEstimator, metaclass=ABCMeta):
    """
    This class is a mixin for classification tasks. It extends the BaseEstimator class and implements additional methods
    specific to classification.

    Methods:
        __init__(**kwargs):
            Initializes the classifier. This method is abstract and must be overridden in subclasses.
        _log_train_progress(epoch_ndx, train_results):
            Logs the training progress.
        _log_validation_progress(epoch_ndx, train_results, val_results):
            Logs the validation progress.
        score(X, y):
            Computes the score of the classifier on the given test data and labels.
        task:
            Returns the task type as a string ('classification').
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the classifier. This method is abstract and must be overridden in subclasses.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """

        super().__init__(**kwargs)

    def _log_train_progress(self, epoch_ndx, train_results):
        """
        Logs the training progress.

        Args:
            epoch_ndx (int): The current epoch index.
            train_results (tuple): A tuple containing the training results.

        Returns:
            None
        """

        train_loss, train_acc, true_labels, pred_labels = train_results
        tqdm.write(f"Epoch {epoch_ndx}, Train Loss: {train_loss} Train F1: {train_acc}", end='\r')

    def _log_validation_progress(self, epoch_ndx, train_results, val_results):
        """
        Logs the validation progress.

        Args:
            epoch_ndx (int): The current epoch index.
            train_results (tuple): A tuple containing the training results.
            val_results (tuple): A tuple containing the validation results.

        Returns:
            None
        """
        
        val_loss, val_acc, true_labels, pred_labels = val_results
        tqdm.write(f"Epoch {epoch_ndx}, Val Loss: {val_loss} Val F1: {val_acc}", end='\r')

    def score(self,
            X: Union[np.ndarray, pd.core.frame.DataFrame],
            y: Union[np.ndarray, pd.core.frame.DataFrame]):
        """
        Computes the score of the classifier on the given test data and labels.

        Args:
            X (Union[np.ndarray, pd.core.frame.DataFrame]): The test data.
            y (Union[np.ndarray, pd.core.frame.DataFrame]): The true labels for the test data.

        Returns:
            true_labels (np.ndarray): The true labels.
            pred_labels (np.ndarray): The predicted labels.
        """

        test_dl = self.dataloader.get_loader(X, y, mode='test')
        _, _, true_labels, pred_labels = self._doValidation(test_dl)
        
        return true_labels, pred_labels

    @property
    def task(self) -> str:
        """
        Returns the task type as a string.

        Returns:
            task (str): The task type ('classification').
        """

        return 'classification'
    
class RegressorMixin(BaseEstimator, metaclass=ABCMeta):
    """
    This class is a mixin for regression tasks. It extends the BaseEstimator class and implements additional methods
    specific to regression.

    Methods:
        __init__(**kwargs):
            Initializes the regressor. This method is abstract and must be overridden in subclasses.
        _log_train_progress(epoch_ndx, train_results):
            Logs the training progress.
        _log_validation_progress(epoch_ndx, train_results, val_results):
            Logs the validation progress.
        predict(X):
            Predicts the targets for the given data.
        task:
            Returns the task type as a string ('regression').
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the regressor. This method is abstract and must be overridden in subclasses.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """

        super().__init__(**kwargs)

    def _log_train_progress(self, epoch_ndx, train_results):
        """
        Logs the training progress.

        Args:
            epoch_ndx (int): The current epoch index.
            train_results (tuple): A tuple containing the training results.

        Returns:
            None
        """

        train_loss, _, _, _ = train_results
        tqdm.write(f"Epoch {epoch_ndx}, Train Loss: {train_loss}", end='\r')

    def _log_validation_progress(self, epoch_ndx, train_results, val_results):
        """
        Logs the validation progress.

        Args:
            epoch_ndx (int): The current epoch index.
            train_results (tuple): A tuple containing the training results.
            val_results (tuple): A tuple containing the validation results.

        Returns:
            None
        """

        train_loss, _, _, _ = train_results
        val_loss, _, _, _ = val_results
        tqdm.write(f"Epoch {epoch_ndx}, Train Loss: {train_loss}, Val Loss: {val_loss}", end='\r')


    def predict(self, 
                X: Union[np.ndarray, pd.core.frame.DataFrame]
                ) -> np.ndarray:
        """
        Computes the prediction of the regressor on the given test data.

        Args:
            X (Union[np.ndarray, pd.core.frame.DataFrame]): The test data.

        Returns:
            output (np.ndarray): The predicted target.
        """

        test_dl = self.dataloader.get_loader(X, y=None, mode='test')
        output = self._predict(test_dl)
        output = self.dataloader.inverse_transform_output(output).cpu().detach().numpy()
        
        return output

    @property
    def task(self) -> str:
        """
        Returns the task type as a string.

        Returns:
            task (str): The task type ('regression').
        """

        return 'regression'
    
class MetaEstimatorMixin:
    _required_parameters = ['model_type', 'task']
    """
    Mixin class for all meta estimators in stockpy.
    """