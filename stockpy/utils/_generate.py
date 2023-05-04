import torch
from pyro.infer import Predictive
import pyro.distributions as dist
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ..config import Config as cfg
from ._model import Model
from typing import Union, Tuple

class Generate(Model):
    def __init__(self,model=None,
                 **kwargs
                 ):
        super().__init__(model=model, **kwargs)

    # Implement the function _generate here to generate mid to long term predictions for each stock
    # using transformers models and reinforcement learning.

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

        output = torch.tensor([]).to(cfg.training.device)
        self._model.eval()

        with torch.no_grad():
            for x_batch, _ in self.train_dl:
                # to device
                x_batch = x_batch.to(cfg.training.device)
                y_star = self._model(x_batch)
                y_star = y_star.squeeze().transpose(0, 1)
                y_star = y_star[-1].squeeze()
                output = torch.cat((output, y_star), 0)

        output.to('cpu')
        output = output.detach().numpy() * self._std() + self._mean()
                    
        return output