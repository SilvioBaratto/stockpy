import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score
from typing import Union, Tuple, Callable
from ._base import BaseComponent

import pyro
from pyro.infer import (
    SVI,
    Trace_ELBO
)
from pyro.optim import ClippedAdam
from tqdm.auto import tqdm
from ._model import Model
from ._dataloader import StockDataset as sd
from abc import ABC, abstractmethod
from ..config import Config as cfg

class Trainer(BaseComponent):
    def __init__(self, model=None, **kwargs):
        super().__init__(model=model, **kwargs)
                
    def _trainRegressor(self,
            train_dl: torch.utils.data.DataLoader,
            val_dl: torch.utils.data.DataLoader) -> None:
        
        best_loss = float('inf')
        counter = 0

        if cfg.training.metrics:
            train_losses = np.zeros(cfg.training.epochs)
        for epoch_ndx in tqdm((range(1, cfg.training.epochs + 1)), position=0, leave=True):
            # calculate training loss
            train_loss = self.component._trainRegressor(train_dl)
            if cfg.training.metrics:
                train_losses[epoch_ndx - 1] = train_loss
            # scheduler
            if cfg.training.scheduler:
                self.component._scheduler.step()
            if epoch_ndx % cfg.training.validation_cadence != 0:
                tqdm.write(f"Epoch {epoch_ndx}, Train Loss: {train_loss}", end='\r')
            else:
                # calculate validation loss
                val_loss = self.component._doValidationRegressor(val_dl)
                tqdm.write(f"Epoch {epoch_ndx}, Val Loss {val_loss}", end='\r')
                # Early stopping
                if cfg.training.early_stopping:
                    stop, best_loss, counter = self.component._earlyStopping(val_loss, best_loss, counter)
                    if stop:
                        break
        
        if cfg.training.metrics:
            train_losses = train_losses[:epoch_ndx]
            return train_losses
        
        return None

    def _trainClassifier(self,
                        train_dl: torch.utils.data.DataLoader,
                        val_dl: torch.utils.data.DataLoader) -> None:

        best_loss = float('inf')
        counter = 0

        if cfg.training.metrics:
            train_losses = np.zeros(cfg.training.epochs)
            train_accuracies = np.zeros(cfg.training.epochs)
            train_f1_scores = np.zeros(cfg.training.epochs)

        for epoch_ndx in tqdm((range(1, cfg.training.epochs + 1)), position=0, leave=True):
            # calculate training loss
            train_loss, train_acc, train_f1 = self.component._trainClassifier(train_dl)
            
            # Update the arrays with the current epoch's metrics
            if cfg.training.metrics:
                train_losses[epoch_ndx - 1] = train_loss
                train_accuracies[epoch_ndx - 1] = train_acc
                train_f1_scores[epoch_ndx - 1] = train_f1

            # scheduler
            if cfg.training.scheduler:
                self.component._scheduler.step()
            if epoch_ndx % cfg.training.validation_cadence != 0:
                tqdm.write(f"Epoch {epoch_ndx}, Train Loss: {train_loss}, Acc {train_acc}, F1_score {train_f1}", end='\r')
            else:
                # calculate validation loss
                val_loss, val_acc, val_f1, _, _ = self.component._doValidationClassifier(val_dl)
                tqdm.write(f"Epoch {epoch_ndx}, Val Loss {val_loss}, Acc {val_acc}, F1_score {val_f1}", end='\r')
                # Early stopping
                if cfg.training.early_stopping:
                    stop, best_loss, counter = self.component._earlyStopping(val_loss, best_loss, counter)
                    if stop:
                        break

        # Trim the arrays in case of early stopping
        if cfg.training.metrics:
            train_losses = train_losses[:epoch_ndx]
            train_accuracies = train_accuracies[:epoch_ndx]
            train_f1_scores = train_f1_scores[:epoch_ndx]

            return train_losses, train_accuracies, train_f1_scores
        
        return None