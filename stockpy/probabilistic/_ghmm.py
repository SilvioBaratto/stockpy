import sys
import datetime
import hashlib
import os
import shutil
import sys
import glob

from os.path import exists

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.infer.autoguide import AutoDiagonalNormal
from torch.autograd import Variable
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import (
    SVI,
    Trace_ELBO
)
from pyro.optim import ClippedAdam
import torch.nn.functional as F

from ..utils import StockDataset, normalize
import pandas as pd
import matplotlib.pyplot as plt

# set style of graphs
plt.style.use('ggplot')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

class Emitter(nn.Module):
    """
    Parameterizes the Gaussian observation likelihood p(y_t | z_t, x_t)
    """
    def __init__(self, z_dim, x_dim, emission_dim, variance):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_x_to_hidden = nn.Linear(x_dim, emission_dim)
        self.lin_hidden_to_mean = nn.Linear(emission_dim, 1)
        # initialize the fixed variance hyperparameter
        self.variance = nn.Parameter(torch.tensor(variance))
        # initialize the non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t, x_t):
        """
        Given the latent z at a particular time step t and the input x at time step t,
        we return the mean and variance of the Gaussian distribution p(y_t|z_t, x_t)
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_x_to_hidden(x_t))
        h3 = h1 + h2  # element-wise sum of the two hidden states
        mean = self.lin_hidden_to_mean(h3)
        return mean, self.variance
    

class GatedTransition(nn.Module):
    """
    Parameterizes the dynamics of the latent variables z_t
    """
    def __init__(self, z_dim, x_dim, transition_dim):
        super().__init__()
        # initialize the two linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_x_to_hidden = nn.Linear(x_dim, transition_dim)
        # initialize the two gated transformations used in the neural network
        self.lin_hidden_to_hidden1 = nn.Linear(transition_dim, transition_dim)
        self.lin_hidden_to_hidden2 = nn.Linear(transition_dim, transition_dim)
        # initialize the non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_t, x_t):
        """
        Given the latent z at a particular time step t and the input x at time step t,
        we return the parameters for the Gaussian distribution p(z_t | z_{t-1}, x_t)
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_x_to_hidden(x_t))
        gated1 = self.sigmoid(self.lin_hidden_to_hidden1(h1))
        gated2 = self.sigmoid(self.lin_hidden_to_hidden2(h2))
        h3 = gated1 * h1 + gated2 * h2
        mu_t = h3
        return mu_t, 1.0

class GHMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Gaussian Hidden Markov Model
    """
    def __init__(self, 
                input_size=4, 
                z_dim=64, 
                emission_dim=64,
                transition_dim=64, 
                variance=0.1,
                pretrained=False
                ):
        
        super().__init__()
        self._z_dim = z_dim
        self._emission_dim = emission_dim
        self._transition_dim = transition_dim

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(z_dim, input_size, emission_dim, variance)
        self.transition = GatedTransition(z_dim, input_size, transition_dim)

        if use_cuda:
            if torch.cuda.device_count() > 1:
                self.emitter = nn.DataParallel(self.emitter)
                self.transition = nn.DataParallel(self.transition)

            self.emitter = self.emitter.to(device)
            self.transition = self.transition.to(device)

        # define a (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(1, z_dim))

    def model(self, x_data, y_data=None, annealing_factor=1.0):
        """
        This function defines the generative model for a Gaussian Hidden Markov Model.
        It generates a sequence of latent variables and observations by looping
        through time steps and sampling from the prior and likelihood.

        Args:
            x_data: the observed data
            y_data: optional, used if the model is being used for prediction
            annealing_factor: optional, used for KL annealing. Defaults to 1.0.
        """
        T_max = x_data.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("ghmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(x_data.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(x_data)):
            # sequentially sample the remaining latent and observed variables
            for t in range(1, T_max):
                # sample the next latent state z_t from the transition model
                z_loc, z_scale = self.transition(z_prev, x_data[:, t - 1, :])
                with poutine.scale(None, annealing_factor):
                    z_t = pyro.sample("z_%d" % t,
                                    dist.Normal(z_loc, z_scale)
                                                .to_event(1))

                # compute the mean of the Gaussian distribution p(y_t | z_t)
                mean_t, variance = self.emitter(z_t, x_data[:, t - 1, :])

                # the next statement instructs pyro to observe y_t according to the
                # Gaussian distribution p(y_t | z_t)
                pyro.sample("obs_y_%d" % t,
                            dist.Normal(mean_t, variance).to_event(1),
                            obs=y_data)
                
                # print(f"model z_prev shape: {z_prev.shape}")
                # print(f"model x_data[:, t - 1, :] shape: {x_data[:, t - 1, :].shape}")   
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t
        
        return z_t

    def guide(self, x_data, y_data=None, annealing_factor=1.0):
        """
        This function defines the guide for a Gaussian Hidden Markov Model.
        It is used to approximate the posterior distribution over the latent variables.

        Args:
            x_data: the observed data
            y_data: optional, used if the model is being used for prediction
            annealing_factor: optional, used for KL annealing. Defaults to 1.0.
        """
        T_max = x_data.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("ghmm", self)

        # initialize the values of the latent variables using heuristics
        z_q_0_expanded = self.z_q_0.expand(x_data.size(0), self._z_dim)
        z_prev = z_q_0_expanded

        with pyro.plate("z_minibatch", len(x_data)):
            # mark z_0 as auxiliary
            for t in range(T_max):
                # define the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.transition(z_prev, x_data[:, t - 1, :])
                z_dist = dist.Normal(z_loc, z_scale).to_event(1)

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(None, annealing_factor):
                    z_t = pyro.sample("z_%d" % t, 
                                      z_dist)
           
                # the latent sampled at this time step will be conditioned
                # upon in the next time step so keep track of it
                z_prev = z_t

            return z_t
    
class GaussianHMM(PyroModule):

    def __init__(self, 
                input_size=4, 
                z_dim=64, 
                emission_dim=64,
                transition_dim=64, 
                variance=0.1,
                pretrained=False
                ):
        # initialize PyroModule
        super(GaussianHMM, self).__init__()

        self._input_size = input_size
        self._z_dim = z_dim
        self._emission_dim = emission_dim
        self._transition_dim = transition_dim
        self._variance = variance
        self._pretrained = pretrained
        self.use_cuda = torch.cuda.is_available()

        # self.model_path = self.__initModelPath()
        self._initModel()
        self.name = "gaussian_hidden_markov_model"

    def _initSVI(self):
        """
        Initializes a Stochastic Variational Inference (SVI) instance to optimize the model and guide.

        Returns:
            svi (pyro.infer.svi.SVI): SVI instance
        """

        return SVI(self._model, 
                  self._guide, 
                  self._initOptimizer(), 
                  loss=Trace_ELBO()
                )
        
    def _initOptimizer(self):
        """
        Initializes the optimizer used to train the model.

        Returns:
            optimizer (pyro.optim.ClippedAdam): Optimizer instance
        """

        adam_params = {"lr": 1e-3, 
                        "betas": (0.96, 0.999),
                        "clip_norm": 10.0, 
                        "lrd": 0.99996,
                        "weight_decay": .0
                    }
        return ClippedAdam(adam_params)
    
    def _initScheduler(self):
        """
        Initializes a learning rate scheduler to control the learning rate during training.

        Returns:
            scheduler (pyro.optim.ExponentialLR): Learning rate scheduler
        """

        return pyro.optim.ExponentialLR({'optimizer': self._optimizer, 
                                         'optim_args': {'lr': 0.01}, 
                                         'gamma': 0.1}
                                         )
        
    def _initTrainDl(self, 
                     x_train, 
                     batch_size, 
                     num_workers, 
                     sequence_length
                     ):
        """
        Initializes the training data loader.

        Parameters:
            x_train (numpy.ndarray or pandas dataset): the training dataset
            batch_size (int): the batch size to use for training
            num_workers (int): the number of workers to use for data loading
            sequence_length (int): the length of the input sequence

        Returns:
            train_dl (torch.utils.data.DataLoader): the training data loader
        """

        train_dl = StockDataset(x_train, sequence_length=sequence_length)

        train_dl = DataLoader(train_dl, 
                            batch_size=batch_size * (torch.cuda.device_count() \
                                                                   if self.use_cuda else 1),  
                            num_workers=num_workers,
                            pin_memory=self.use_cuda,
                            shuffle=False
                            )

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._sequence_length = sequence_length

        return train_dl

    def _initValDl(self, 
                   x_test
                   ):
        """
        Initializes the validation data loader.

        Parameters:
            x_test (numpy.ndarray or pandas dataset): the validation dataset

        Returns:
            val_dl (torch.utils.data.DataLoader): the validation data loader
        """

        val_dl = StockDataset(x_test, 
                                sequence_length=self._sequence_length
                                )

        val_dl = DataLoader(val_dl, 
                            batch_size=self._batch_size * (torch.cuda.device_count() \
                                                    if self.use_cuda else 1), 
                            num_workers=self._num_workers,
                            pin_memory=self.use_cuda,
                            shuffle=False
                            )
        
        return val_dl
    
    def _initTrainValData(self, 
                          x_train,
                          validation_sequence,
                          batch_size,
                          num_workers,
                          sequence_length
                          ):
        """
        Initializes the training and validation data loaders.

        Parameters:
            x_train (numpy.ndarray): the training dataset
            validation_sequence (int): the number of time steps to reserve for validation during training
            batch_size (int): the batch size to use during training
            num_workers (int): the number of workers to use for data loading
            sequence_length (int): the length of the input sequence

        Returns:
            train_dl (torch.utils.data.DataLoader): the training data loader
            val_dl (torch.utils.data.DataLoader): the validation data loader
        """
                
        scaler = normalize(x_train)

        x_train = scaler.fit_transform()
        val_dl = x_train[-validation_sequence:]
        x_train = x_train[:len(x_train)-len(val_dl)]

        train_dl = self._initTrainDl(x_train, 
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        sequence_length=sequence_length
                                        )

        val_dl = self._initValDl(val_dl)

        return train_dl, val_dl

    def fit(self, 
            x_train,
            epochs=10,
            sequence_length=30,
            batch_size=8, 
            num_workers=4,
            validation_sequence=30, 
            validation_cadence=5,
            patience=5
            ):
        """
        Fits the neural network model to a given dataset.

        Parameters:
            x_train (numpy.ndarray): the training dataset
            epochs (int): the number of epochs to train the model for
            sequence_length (int): the length of the input sequence
            batch_size (int): the batch size to use during training
            num_workers (int): the number of workers to use for data loading
            validation_sequence (int): the number of time steps to reserve for validation during training
            validation_cadence (int): how often to run validation during training
            patience (int): how many epochs to wait for improvement in validation loss before stopping early

        Returns:
            None
        """

        train_dl, val_dl = self._initTrainValData(x_train,
                                                  validation_sequence,
                                                  batch_size,
                                                  num_workers,
                                                  sequence_length
                                                  )
        
        self._train(epochs,
                    train_dl,
                    val_dl,
                    validation_cadence,
                    patience
                    )
          
    def _train(self, 
               epochs,
               train_dl,
               val_dl,
               validation_cadence,
               patience
               ):
        """
        Trains the neural network model on the training dataset.

        Parameters:
            epochs (int): the number of epochs to train the model for
            train_dl (torch.utils.data.DataLoader): the training data loader
            val_dl (torch.utils.data.DataLoader): the validation data loader
            validation_cadence (int): how often to run validation during training
            patience (int): how many epochs to wait for improvement in validation loss before stopping early

        Returns:
            None
        """

        best_loss = float('inf')
        counter = 0

        for epoch_ndx in tqdm((range(1, epochs + 1)), position=0, leave=True):
            epoch_loss = 0.0
            for x_batch, y_batch in train_dl:  
                loss = self._computeBatchLoss(x_batch, y_batch)
                epoch_loss += loss
            
            self._scheduler.step()

            if epoch_ndx % validation_cadence != 0:                
                print(f"Epoch {epoch_ndx}, Loss: {epoch_loss / len(train_dl)}")

            else:
                total_loss = self._doValidation(val_dl)

                print(f"Epoch {epoch_ndx}, Val Loss {total_loss / len(val_dl)}")

                # Early stopping
                stop, best_loss, counter = self._earlyStopping(total_loss, 
                                                               best_loss, 
                                                               counter, 
                                                               patience,
                                                               epoch_ndx
                                                               )
                if stop:
                    break


    def _computeBatchLoss(self,
                          x_batch,
                          y_batch
                          ):    
        """
        Computes the loss for a given batch of data.

        Parameters:
            x_batch (torch.Tensor): the input data
            y_batch (torch.Tensor): the target data

        Returns:
            torch.Tensor: the loss for the given batch of data
        """  

        loss = self._svi.step(
            x_data=x_batch,
            y_data=y_batch
        )
        # keep track of the training loss
        return loss  # This is the loss over the entire batch
        
    def _doValidation(self, 
                      val_dl
                      ):
        """
        Performs validation on a given validation data loader.

        Parameters:
            val_dl (torch.utils.data.DataLoader): the validation data loader

        Returns:
            float: the total loss over the validation set
        """

        total_loss = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                loss = self._svi.evaluate_loss(x_batch, y_batch) 
                total_loss += loss

        return total_loss 
    
    def _earlyStopping(self,
                       total_loss,
                       best_loss,
                       counter,
                       patience,
                       epoch_ndx
                       ):
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
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch_ndx = epoch_ndx
            self._saveModel('ghmm', best_epoch_ndx)
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"No improvement after {patience} epochs. Stopping early.")
            return True, best_loss, counter
        else:
            return False, best_loss, counter

    def predict(self, x_test):
        """
        Make predictions on a given test set.

        Parameters:
            x_test (np.ndarray): the test set to make predictions on

        Returns:
            np.ndarray: the predicted values for the given test set
        """

        # set the model to evaluation mode
        self._model.eval()

        # transform x_test in data loader
        scaler = normalize(x_test)
        x_test = scaler.fit_transform()
        self.std = scaler.std()
        self.mean = scaler.mean()

        val_dl = self._initValDl(x_test)

        # create a list to hold the predicted y values
        predicted_y = []

        # iterate over the test data in batches
        for x_batch, _ in val_dl:
            # make predictions for the current batch
            with torch.no_grad():
                # compute the mean of the emission distribution for each time step
                *_, z_loc, z_scale = self._model.guide(x_batch)
                z_scale = F.softplus(z_scale)
                z_t = dist.Normal(z_loc, z_scale).rsample()
                mean_t, _ = self._model.emitter(z_t, x_batch)
                
                # get the mean for the last time step
                mean_last = mean_t[:, -1, :]

            # add the predicted y values for the current batch to the list
            predicted_y.append(mean_last * self.std + self.mean)

        # concatenate the predicted y values for all batches into a single tensor
        predicted_y = torch.cat(predicted_y)

        # reshape the tensor to get an array of shape [151,1]
        predicted_y = predicted_y.reshape(-1, 1)

        # return the predicted y values as a numpy array
        return predicted_y.numpy()
    
    def _initModel(self):
        """
        Initializes the neural network model.

        Returns:
            None
        """

        model = GHMM(
                input_size=self._input_size, 
                z_dim=self._z_dim, 
                emission_dim=self._emission_dim,
                transition_dim=self._transition_dim, 
                variance=self._variance
            )
        if self._pretrained:
            path = self._initModelPath('dmm')
            model_dict = torch.load(path)
            model.load_state_dict(model_dict['model_state'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            self._model = model.to(device)

        self._model = model.model
        self._guide = model.guide
        
        pyro.clear_param_store()
        self._optimizer = self._initOptimizer()
        self._svi = self._initSVI()
        # Create learning rate scheduler
        self._scheduler = self._initScheduler()

    def _saveModel(self, type_str, epoch_ndx):
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
            'GHMM',
            '{}_{}_{}_{}_{}_{}.state'.format(
                    type_str,
                    self._input_size,
                    self._z_dim,
                    self._emission_dim,
                    self._transition_dim,
                    self._variance
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self._model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self._optimizer.get_state(),
            'optimizer_name': type(self._optimizer).__name__,
            'epoch': epoch_ndx
        }

        torch.save(state, file_path)

        with open(file_path, 'rb') as f:
            hashlib.sha1(f.read()).hexdigest()

    def _initModelPath(self, type_str):
        """
        Initializes the model path.

        Parameters:
            type_str (str): a string indicating the type of model

        Returns:
            str: the path to the initialized model
        """

        model_dir = '../../models/BNN'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        local_path = os.path.join(
            '..', 
            '..', 
            'models', 
            'GHMM', 
            type_str + '_{}_{}_{}_{}_{}.state'.format(self._input_size,
                                                            self._z_dim,
                                                            self._emission_dim,
                                                            self._transition_dim,
                                                            self._variance
                                                            ),
            )

        file_list = glob.glob(local_path)
        
        if not file_list:
            raise ValueError(f"No matching model found in {local_path} for the given parameters.")
        
        # Return the most recent matching file
        return file_list[0]
    
    def trading(self, 
                predicted, 
                real, 
                shares=0, 
                stop_loss=0.0, 
                initial_balance=10000, 
                threshold=0.0, 
                plot=True
                ):
        """
        Simulate trading based on predicted and real stock prices.

        Args:
            predicted (np.ndarray): Array of predicted stock prices.
            real (np.ndarray): Array of real stock prices.
            shares (int): Number of shares held at the start of the simulation. Default is 0.
            stop_loss (float): Stop loss percentage. If the stock price falls below this percentage of the initial price,
                            all shares will be sold. Default is 0.0.
            initial_balance (float): Initial balance to start trading with. Default is 10000.
            threshold (float): Buy/Sell threshold. Default is 0.0.
            plot (bool): Whether to plot the trading simulation or not. Default is True.

        Returns:
            tuple: A tuple containing balance (float), total profit/loss (float), percentage increase (float), 
            and transactions (list of tuples). The transactions are of the form (timestamp, price, action, shares, balance).
        """

        assert predicted.shape == real.shape, "predicted and real must have the same shape"
        assert shares >= 0, "shares cannot be negative"
        assert initial_balance >= 0, "initial_balance cannot be negative"
        assert 0 <= stop_loss <= 1, "stop_loss must be between 0 and 1"

        transactions = []
        balance = initial_balance
        num_shares = shares
        total_profit_loss = 0

        if num_shares == 0 and balance >= real[0]:
            num_shares = int(balance / real[0])
            balance -= num_shares * real[0]
            transactions.append((0, real[0], "BUY", num_shares, balance))

        for i in range(1, len(predicted)):
            if predicted[i] > real[i-1] * (1 + threshold):
                if num_shares == 0:
                    num_shares = int(balance / real[i])
                    balance -= num_shares * real[i]
                    transactions.append((i, real[i], "BUY", num_shares, balance))
                elif num_shares > 0:
                    balance += num_shares * real[i]
                    total_profit_loss += (real[i] - real[i-1]) * num_shares
                    transactions.append((i, real[i], "SELL", num_shares, balance))
                    num_shares = 0
            elif predicted[i] < real[i-1] * (1 - threshold):
                if num_shares == 0:
                    continue
                elif num_shares > 0:
                    balance += num_shares * real[i]
                    total_profit_loss += (real[i] - real[i-1]) * num_shares
                    transactions.append((i, real[i], "SELL", num_shares, balance))
                    num_shares = 0

            if stop_loss > 0 and num_shares > 0 and real[i] < (real[0] - stop_loss):
                balance += num_shares * real[i]
                total_profit_loss += (real[i] - real[i-1]) * num_shares
                transactions.append((i, real[i], "SELL", num_shares, balance))
                num_shares = 0

        if num_shares > 0:
            balance += num_shares * real[-1]
            total_profit_loss += (real[-1] - real[-2]) * num_shares
            transactions.append((len(predicted)-1, real[-1], "SELL", num_shares, balance))
            num_shares = 0

        percentage_increase = (balance - initial_balance) / initial_balance * 100

        if plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(real, label='Real')
            ax.plot(predicted, label='Predicted')
            buy_scatter = ax.scatter([], [], c='g', marker='^', s=100)
            sell_scatter = ax.scatter([], [], c='r', marker='v', s=100)
            for transaction in transactions:
                timestamp, price, action, shares, balance = transaction
                if action == 'BUY':
                    buy_scatter = ax.scatter(timestamp, predicted[timestamp], c='g', marker='^', s=100)
                elif action == 'SELL':
                    sell_scatter = ax.scatter(timestamp, predicted[timestamp], c='r', marker='v', s=100)
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.set_title('Trading Simulation')
            fig.autofmt_xdate()
            ax.legend((ax.plot([], label='Real')[0], ax.plot([], label='Predicted')[0], buy_scatter, sell_scatter),
                    ('Real', 'Predicted', 'Buy', 'Sell'))
            ax.text(0.05, 0.05, 
                    'Percentage increase: ${:.2f}%'.format(percentage_increase[0]), 
                    ha='left', va='center',
                      transform=ax.transAxes, 
                      bbox=dict(facecolor='white', alpha=0.5)
                      )
            plt.show()
        
        return balance, total_profit_loss, percentage_increase, transactions
