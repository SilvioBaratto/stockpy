import torch
from pyro.infer import Predictive
import pyro.distributions as dist
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ._dataloader import StockDataset as sd
from ._model import Model
from typing import Union, Tuple

import pyro
from pyro.infer import (
    SVI,
    Trace_ELBO
)
from pyro.optim import ClippedAdam
from tqdm.auto import tqdm
from ._model import Model
from ._dataloader import StockDataset as sd
from ._base import BaseComponent
from ..config import Config as cfg

class Predict(BaseComponent):
    def __init__(self, model=None, **kwargs):
        super().__init__(model=model, **kwargs)

    def _predict(self,
                test_dl : torch.utils.data.DataLoader
                ) -> torch.Tensor:
        """
        Generate predictions for the given test set using the trained model.

        This method first normalizes the input test set using the same normalization method as the training set. It then
        initializes a validation DataLoader and generates predictions using either a probabilistic model (e.g., Bayesian
        neural network) or a neural network model (e.g., BiGRU), depending on the model type. The predicted values are
        rescaled back to the original scale and returned as a NumPy array.

        Parameters:
            x_test (Union[np.ndarray, pd.core.frame.DataFrame]): The test set to make predictions on, either as a NumPy array or pandas DataFrame.

        Returns:
            np.ndarray: The predicted target values for the given test set, as a NumPy array.
        """

        return self.component._predict(test_dl)

    def _score(self,
                test_dl : torch.utils.data.DataLoader
                ) -> np.ndarray:
        
        # return self.component._predict(test_dl).detach().numpy()
        return self.component._score(test_dl)