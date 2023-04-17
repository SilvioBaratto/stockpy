import torch
from pyro.infer import Predictive
import pyro.distributions as dist
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ..config import training
from ._dataloader import normalize
from ._model import Model
from typing import Union, Tuple

class Generate(Model):
    def __init__(self,
                 model=None,
                 **kwargs
                 ):
        super().__init__(model=model, **kwargs)

    def _generate(self,
                  n_samples : int
                  ) -> np.ndarray:
        """
        Generate mid to long term prediction

        Parameters:
            n_samples (int): number of samples for the long term prediction

        Return:
            np.ndarray: The predicted long term forecasting.
        """
        
        # TODO in this function I want to generate mid to long term predictions for each stock 
        # using transformers models and reinforcement learning. 

        raise NotImplementedError("This method is not implemented yet.")
