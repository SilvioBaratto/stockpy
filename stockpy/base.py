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

    @abstractmethod
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(cfg.nn, key, value)
            setattr(cfg.prob, key, value)

    @abstractmethod
    def forward(self, x):
        # Your code here
        pass  

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def _initOptimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def _initScheduler(self) -> torch.optim.lr_scheduler.StepLR:
        pass

    @abstractmethod
    def _initComponent(self):
        pass

    @abstractmethod
    def _doTraining(self):
        pass

    @abstractmethod
    def _doValidation(self):
        pass

    def fit(self, 
            X: Union[np.ndarray, pd.core.frame.DataFrame],
            y: Union[np.ndarray, pd.core.frame.DataFrame],
            **kwargs
            ) -> None:
        
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

        print(self.eval())

        if cfg.training.pretrained:
            self._loadModel()

        return self._train(train_dl, val_dl)

    def _train(self,
            train_dl: torch.utils.data.DataLoader,
            val_dl: torch.utils.data.DataLoader) -> None:
        
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
        Saves the model to the specified directory.

        :param epoch: The current epoch.
        :type epoch: int
        :param loss: The current loss.
        :type loss: float
        :param save_dir: The directory to save the model to.
        :type save_dir: str
        """
        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(lib_dir, 'save', file_format.format(*args))
        
        if cfg.training.folder is None:
            lib_dir = os.path.dirname(os.path.abspath(__file__))  # directory of the library
        else:
            lib_dir = cfg.training.folder

        file_path_configs = self._log_build_file_path()

        file_path = build_file_path(file_path_configs["file_format"],
                                    *file_path_configs["args"])
                
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        state = self._log_model_state()

        torch.save(state, file_path)

    def _loadModel(self) -> None:
        """
        Loads the model from the specified directory.
        """
        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(lib_dir, 'save', file_format.format(*args))

        if cfg.training.folder is None:
            lib_dir = os.path.dirname(os.path.abspath(__file__))  # directory of the library
        else:
            lib_dir = cfg.training.folder

        file_path_configs = self._log_build_file_path()

        file_path = build_file_path(file_path_configs["file_format"],
                                    *file_path_configs["args"])

        if not os.path.exists(file_path):
            raise ValueError(f"No matching model found in {file_path} for the given parameters.")

        state = torch.load(file_path)

        # Only load the model_state part of the dictionary
        model_state = state['model_state']

        self.load_state_dict(model_state)

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

        :param device: The device to move the model to.
        :type device: torch.device
        """
        super().to(device)
    
    @property
    def name(self):
        return self.__class__.__name__

class ClassifierMixin(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _log_train_progress(self, epoch_ndx, train_results):
        train_loss, train_acc, true_labels, pred_labels = train_results
        tqdm.write(f"Epoch {epoch_ndx}, Train Loss: {train_loss} Train F1: {train_acc}", end='\r')

    def _log_validation_progress(self, epoch_ndx, train_results, val_results):
        val_loss, val_acc, true_labels, pred_labels = val_results
        tqdm.write(f"Epoch {epoch_ndx}, Val Loss: {val_loss} Val F1: {val_acc}", end='\r')

    def score(self,
              X: Union[np.ndarray, pd.core.frame.DataFrame],
              y: Union[np.ndarray, pd.core.frame.DataFrame]):
        
        test_dl = self.dataloader.get_loader(X, y, mode = 'test')
        _, _, true_labels, pred_labels = self._doValidation(test_dl)
        
        return true_labels, pred_labels
    
    @property
    def task(self) -> str:
        return 'classification'
    
class RegressorMixin(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _log_train_progress(self, epoch_ndx, train_results):
        train_loss, _, _, _ = train_results
        tqdm.write(f"Epoch {epoch_ndx}, Train Loss: {train_loss}", end='\r')

    def _log_validation_progress(self, epoch_ndx, train_results, val_results):
        train_loss, _, _, _ = train_results
        val_loss, _, _, _ = val_results
        tqdm.write(f"Epoch {epoch_ndx}, Train Loss: {train_loss}, Val Loss: {val_loss}", end='\r')

    
    def predict(self, 
                X: Union[np.ndarray, pd.core.frame.DataFrame]
                ) -> np.ndarray:

        test_dl = self.dataloader.get_loader(X, y=None, mode = 'test')
        output = self._predict(test_dl)
        output = self.dataloader.inverse_transform_output(output).cpu().detach().numpy()
        
        return output
    
    @property
    def task(self) -> str:
        return 'regression'