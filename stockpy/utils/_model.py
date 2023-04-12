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
from ..config import nn_args, prob_args, shared, training

class Model():
    def __init__(self,
                 model=None,
                 **kwargs
                 ):
        
        if model.model_type == "neural_network":
            for key, value in kwargs.items():
                setattr(nn_args, key, value)
        elif model.model_type == "probabilistic":
            for key, value in kwargs.items():
                setattr(prob_args, key, value)        

        self._initModel(model)

    def _modelEval(self):
        print(self._model.eval())
    
    def _initModel(self, 
                   model : Union[nn.Module, PyroModule]
                   ) -> None:
        """
        Initializes the neural network model.
        Returns:
            None
        """
        self.name = model.__class__.__name__
        
        if shared.pretrained:
            path = self._initModelPath(model, self.name)
            model_dict = torch.load(path)
            model.load_state_dict(model_dict['model_state'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if training.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            self._model = model.to(device)

        self._model = model

        if self._model.model_type == "neural_network":
            self.type = 'neural_network'
        elif self._model.model_type == "probabilistic":
            self.type = 'probabilistic'
            if self.name == 'BayesianNN':
                self._guide = AutoDiagonalNormal(self._model)
            else:
                self._guide = self._model.guide
                    
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
            self.name, 
            type_str + '_{}_{}.state'.format(shared.dropout,
                                                shared.weight_decay
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

        model_dir = '../../models/' + self.name
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        local_path = os.path.join(
            '..', 
            '..', 
            'models', 
            self.name, 
            type_str + '_{}_{}.state'.format(shared.dropout,
                                                shared.weight_decay
                                                ),
            )

        file_list = glob.glob(local_path)
        
        if not file_list:
            raise ValueError(f"No matching model found in {local_path} for the given parameters.")
        
        # Return the most recent matching file
        return file_list[0]