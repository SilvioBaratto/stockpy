import datetime
import hashlib
import os
import shutil
import sys
import glob
import sys

import torch
import torch.nn as nn
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
from ..config import Config as cfg

class Model:
    def __init__(self,
                 model=None,
                 **kwargs
                 ):
        
        attr_settings = {
            "neural_network": cfg.nn, 
            "probabilistic": cfg.prob,
        }

        model_type = model.model_type
        if model_type in attr_settings:
            for key, value in kwargs.items():
                setattr(attr_settings[model_type], key, value)

        else:
            raise ValueError("Model type not recognized")
    
    def _initModel(self, 
                model: Union[nn.Module, PyroModule]
                ) -> None:
        """
        Initializes the neural network model.
        Returns:
            None
        """
        self._model = model
        self._model.to(cfg.training.device)
        
        if cfg.training.pretrained:
            path = self._initModelPath()
            print(f"Loading model from {path}")
            model_dict = torch.load(path)
            self._model.load_state_dict(model_dict['model_state'])

        if cfg.training.use_cuda:
            if torch.cuda.device_count() > 1:
                self._model = nn.DataParallel(self._model)

        model_type_map = {
            "neural_network": "neural_network",
            "probabilistic": "probabilistic"
        }

        self.type = model_type_map.get(self._model.model_type)
        self.name = self._model.name

        if self.type == "probabilistic":
            allowed_names = ['BayesianNNRegressor', 'BayesianNNClassifier', 
                             'BayesianCNNRegressor', 'BayesianCNNClassifier']
            if self._model.name in allowed_names:
                self._guide = AutoDiagonalNormal(self._model)
            else:
                self._guide = self._model.guide
                    
    def _saveModel(self, 
                type_str: str,
                optimizer: torch.optim.Optimizer,
                ) -> None:
        """
        Saves the model to disk.
        Parameters:
            type_str (str): a string indicating the type of model
        Returns:
            None
        """

        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(lib_dir, 'save', self.type, self._model.name, file_format.format(*args))

        if cfg.training.folder is None:
            lib_dir = os.path.dirname(os.path.abspath(__file__))  # directory of the library
        else:
            lib_dir = cfg.training.folder

        file_path_configs = {
            "probabilistic": {
                "file_format": type_str + '_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.state',
            "args": (self._model.input_size, cfg.prob.hidden_size, self._model.output_size,
                    cfg.prob.rnn_dim, cfg.prob.z_dim, cfg.prob.emission_dim,
                    cfg.prob.transition_dim, cfg.prob.variance, cfg.comm.dropout, 
                    cfg.training.lr, cfg.training.weight_decay)
            },
            "neural_network": {
                "file_format": type_str + '_{}_{}_{}_{}_{}_{}_{}.state',
                "args": (self._model.input_size, cfg.nn.hidden_size, self._model.output_size,
                        cfg.nn.num_layers, cfg.comm.dropout, cfg.training.lr, cfg.training.weight_decay)
            },
            "generative": {
                "file_format": type_str + '_{}_{}_{}_{}_{}_{}.state',
                "args": (self._model.input_size, cfg.nn.hidden_size, cfg.nn.num_layers,
                        cfg.comm.dropout, cfg.training.lr, cfg.training.weight_decay)
            }
        }

        file_path = build_file_path(file_path_configs[self.type]["file_format"],
                                    *file_path_configs[self.type]["args"])

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        if isinstance(self._model, torch.nn.DataParallel):
            self._model = self._model.module

        state = {
            'model_state': self._model.state_dict(),
            'model_name': type(self._model).__name__,
            'optimizer_state': optimizer.state_dict() if self.type  \
                                != "probabilistic" else optimizer.get_state(),
            'optimizer_name': type(optimizer).__name__,
        }

        torch.save(state, file_path)

    def _initModelPath(self) -> str:
        """
        Initializes the model path.
        Returns:
            str: the path to the initialized model
        """

        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(model_dir, file_format.format(*args))

        if cfg.training.folder is None:
            lib_dir = os.path.dirname(os.path.abspath(__file__))  # directory of the library
        else:
            lib_dir = cfg.training.folder

        model_dir = os.path.join(lib_dir, 'save', self._model.model_type, self._model.name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        file_path_configs = {
            "probabilistic": {
                "file_format": self._model.name + '_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.state',
                "args": (self._model.input_size, cfg.prob.hidden_size, self._model.output_size,
                        cfg.prob.rnn_dim, cfg.prob.z_dim, cfg.prob.emission_dim,
                        cfg.prob.transition_dim, cfg.prob.variance, cfg.comm.dropout, 
                        cfg.training.lr, cfg.training.weight_decay)
            },
            "neural_network": {
                "file_format": self._model.name + '_{}_{}_{}_{}_{}_{}_{}.state',
                "args": (self._model.input_size, cfg.nn.hidden_size, self._model.output_size,
                        cfg.nn.num_layers, cfg.comm.dropout, cfg.training.lr, cfg.training.weight_decay)
            },
            "generative": {
                "file_format": self._model.name + '_{}_{}_{}_{}_{}_{}.state',
                "args": (self._model.input_size, cfg.nn.hidden_size, cfg.nn.num_layers,
                        cfg.comm.dropout, cfg.training.lr, cfg.training.weight_decay)
            }
        }

        local_path = build_file_path(file_path_configs[self._model.model_type]["file_format"],
                                    *file_path_configs[self._model.model_type]["args"])

        file_list = glob.glob(local_path)

        if not file_list:
            raise ValueError(f"No matching model found in {local_path} for the given parameters.")

        # Return the most recent matching file
        return file_list[0]