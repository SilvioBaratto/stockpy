import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score
from typing import Union, Tuple, Callable

import pyro
from pyro.infer import (
    SVI,
    Trace_ELBO
)
from pyro.optim import ClippedAdam
from tqdm.auto import tqdm
from ._model import Model
from ._dataloader import StockDataset as sd
from ._probabilistic import Probabilistic
from ._neural_network import NeuralNetwork
from abc import ABC, abstractmethod
from ..config import Config as cfg

class BaseComponent(ABC):
    def __init__(self, model=None, **kwargs):
        for key, value in kwargs.items():
            setattr(cfg.nn, key, value)
            setattr(cfg.prob, key, value)

    def _initModel(self, input_size: int, output_size: int, **kwargs):
        model = self._create_model(input_size=input_size, output_size=output_size)
        component = {
            "probabilistic": Probabilistic,
            "neural_network": NeuralNetwork
        }

        model_type = model.model_type
        if model_type not in component:
            raise ValueError(f"Unsupported component type: {model_type}")

        self.component = component[model_type](model=model, **kwargs)
        self.model = model
        self.name = model.name
        self.type = model_type
        # self.category = model.category


    @abstractmethod
    def _create_model(self, input_size: int, output_size: int):
        pass