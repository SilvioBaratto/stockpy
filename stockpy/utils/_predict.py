import torch
from pyro.infer import Predictive
import pyro.distributions as dist
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ..config import training
from ._dataloader import normalize
from ._dataloader import _initValDl
from ._model import Model
from typing import Union, Tuple

class Predict(Model):
    def __init__(self, 
                 model = None,
                 **kwargs
                 ):
        super().__init__(model, **kwargs)

    def _predict(self, 
                x_test: Union[np.ndarray, pd.core.frame.DataFrame]
                ) -> np.ndarray:
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
        
        scaler = normalize(x_test)
        x_test = scaler.fit_transform()
        val_dl = _initValDl(x_test)

        if self.type == "probabilistic":
            output = self._predict_probabilistic(val_dl)
        elif self.type == "neural_network":
            output = self._predict_neural_network(val_dl)
        else:
            raise ValueError("Model type not recognized")

        output = output.detach().numpy() * scaler.std() + scaler.mean()
                    
        return output

    def _predict_neural_network(self, 
                                val_dl : torch.utils.data.DataLoader
                                ) -> torch.Tensor:
        """
        Predict target values for the given validation DataLoader using a neural network model.

        This method generates predictions using a neural network model (e.g., BiGRU). The method sets the model to evaluation
        mode, iterates over the input data in batches, and generates predictions for each batch. The predictions are then
        concatenated into a single output tensor.

        Parameters:
            val_dl (torch.utils.data.DataLoader): The validation DataLoader containing the input data to make predictions on.

        Returns:
            torch.Tensor: The predicted target values as a torch.Tensor.
        """
        output = torch.tensor([])
        self._model.eval()
        
        with torch.no_grad():
            for x_batch, _ in val_dl:
                y_star = self._model(x_batch)
                output = torch.cat((output, y_star), 0)
                
        return output
    
    def _predict_probabilistic(self,
                               val_dl : torch.utils.data.DataLoader
                               ) -> torch.Tensor:
        """
        Predict target values for the given validation DataLoader using a probabilistic model.

        This method generates predictions using a Bayesian Neural Network or a Deep Markov Model. For the Bayesian Neural
        Network, it uses the Pyro Predictive class to generate samples and compute the mean of the predicted values. For the
        Deep Markov Model, it computes the mean of the emission distribution for each time step and retrieves the mean for
        the last time step.

        Parameters:
            val_dl (torch.utils.data.DataLoader): The validation DataLoader containing the input data to make predictions on.

        Returns:
            torch.Tensor: The predicted target values as a torch.Tensor.
        """     
        if self.name == 'BayesianNN':
            output = torch.tensor([])
            for x_batch, _ in val_dl:
                predictive = Predictive(model=self._model, 
                                        guide=self._guide, 
                                        num_samples=training.batch_size,
                                        return_sites=("linear.weight", 
                                                        "obs", 
                                                        "_RETURN")
                                                    )
                samples = predictive(x_batch)
                site_stats = {}
                for k, v in samples.items():
                    site_stats[k] = {
                        "mean": torch.mean(v, 0)
                    }

                y_pred = site_stats['_RETURN']['mean']
                output = torch.cat((output, y_pred), 0)
            
            return output
        
        else: 
            # create a list to hold the predicted y values
            output = []

            # iterate over the test data in batches
            for x_batch, _ in val_dl:
                # make predictions for the current batch
                with torch.no_grad():
                    # compute the mean of the emission distribution for each time step
                    *_, z_loc, z_scale = self._guide(x_batch)
                    z_scale = F.softplus(z_scale)
                    z_t = dist.Normal(z_loc, z_scale).rsample()
                    mean_t, _ = self._model.emitter(z_t, x_batch)
                    
                    # get the mean for the last time step
                    mean_last = mean_t[:, -1, :]

                # add the predicted y values for the current batch to the list
                output.append(mean_last)

            # concatenate the predicted y values for all batches into a single tensor
            output = torch.cat(output)

            # reshape the tensor to get an array of shape [151,1]
            output = output.reshape(-1, 1)

            # return the predicted y values as a numpy array
            return output