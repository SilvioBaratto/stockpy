import sys
import datetime
import hashlib
import os
import shutil
import sys
import glob
sys.path.append("../")

import argparse
import logging
import time
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
from torch.autograd import Variable
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    config_enumerate,
    TraceMeanField_ELBO,
    Predictive
)
from pyro.optim import ClippedAdam
import torch.nn.functional as F

from utils import StockDataset, normalize
import pandas as pd
import matplotlib.pyplot as plt

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
    
class Combiner(nn.Module):
    """
    Combines the previous hidden state z_{t-1} and the current input x_t
    to produce the hidden state h_t, which is used by the emitter and
    transition networks.
    """
    def __init__(self, z_dim, rnn_dim, hidden_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.lin_rnn_to_hidden = nn.Linear(rnn_dim, hidden_dim)
        self.hidden_to_loc = nn.Linear(hidden_dim, z_dim)
        self.hidden_to_scale = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, z_prev, rnn_input):
        hidden = self.relu(self.lin_z_to_hidden(z_prev) + self.lin_rnn_to_hidden(rnn_input))
        loc = self.hidden_to_loc(hidden)
        scale = F.softplus(self.hidden_to_scale(hidden))
        return loc, scale
    
class DeepMarkovModel(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """
    def __init__(self, 
                input_dim=4, 
                z_dim=64, 
                emission_dim=64,
                transition_dim=64, 
                rnn_dim=32, 
                rnn_dropout_rate=0.1,
                variance=0.1
                ):
        
        super().__init__()
        self._z_dim = z_dim
        self._emission_dim = emission_dim
        self._transition_dim = transition_dim
        self._rnn_dim = rnn_dim

        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(z_dim, input_dim, emission_dim, variance)
        self.transition = GatedTransition(z_dim, input_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim, emission_dim)
        self.rnn = nn.GRU(input_size=input_dim, 
                          hidden_size=rnn_dim,
                          # nonlinearity='relu', 
                          batch_first=True,
                          bidirectional=False, 
                          num_layers=2, 
                          dropout=rnn_dropout_rate
                          )

        # define a (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(1, z_dim))
        # define a (trainable) parameter for the initial hidden state of the RNN
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

    def model(self, x_data, y_data=None, annealing_factor=1.0):

        # this is the number of time steps we need to process in the data
        T_max = x_data.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(x_data.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(x_data)):
            # sample the latents z and observed y's one time step at a time
            for t in range(1, T_max + 1):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing.

                # first compute the parameters of the diagonal gaussian
                # distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.transition(z_prev, x_data[:, t - 1, :])

                # then sample z_t according to dist.Normal(z_loc, z_scale).
                # note that we use the reshape method so that the univariate
                # Normal distribution is treated as a multivariate Normal
                # distribution with a diagonal covariance.
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
        # this is the number of time steps we need to process in the mini-batch
        T_max = x_data.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)
        
        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(2, x_data.size(0), self.rnn.hidden_size).contiguous()
        
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(x_data, h_0_contig)
        
        z_q_0_expanded = self.z_q_0.expand(x_data.size(0), self._z_dim)
        z_prev = z_q_0_expanded
        
        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(x_data)):
            # sample the latents z one time step at a time
            for t in range(1, T_max + 1):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})

                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                z_dist = dist.Normal(z_loc, z_scale).to_event(1)

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(None, annealing_factor):
                    z_t = pyro.sample("z_%d" % t, 
                                      z_dist)

                # print(f"guide z_prev shape: {z_prev.shape}")
                # print(f"guide rnn_output[:, t - 1, :] shape: {rnn_output[:, t - 1, :].shape}")            
                # the latent sampled at this time step will be conditioned
                # upon in the next time step so keep track of it
                z_prev = z_t

            return z_t
    
class DMM(PyroModule):

    def __init__(self, 
                input_dim=4, 
                z_dim=64, 
                emission_dim=64,
                transition_dim=64, 
                rnn_dim=32, 
                rnn_dropout_rate=0.0,
                variance=0.1,
                pretrained=False
                ):
        # initialize PyroModule
        super(DMM, self).__init__()

        self._input_dim = input_dim
        self._z_dim = z_dim
        self._emission_dim = emission_dim
        self._transition_dim = transition_dim
        self._rnn_dim = rnn_dim
        self._rnn_dropout_rate = rnn_dropout_rate
        self._variance = variance
        self._pretrained = pretrained
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        # self.model_path = self.__initModelPath()
        self.dmm = self.__initModel()
        self.optimizer = self.__initOptimizer()
        self.step_model = self.__initStepModel()
        self.step_guide = self.__initGuide()

        self.name = "deep_markov_network"

    def __initModel(self):

        if self._pretrained:
            model_dict = torch.load(self.model_path)

            dmm = DeepMarkovModel(
                input_dim=self._input_dim, 
                z_dim=self._z_dim, 
                emission_dim=self._emission_dim,
                transition_dim=self._transition_dim, 
                rnn_dim=self._rnn_dim, 
                rnn_dropout_rate=self._rnn_dropout_rate,
                variance=self._variance
            )

            dmm.load_state_dict(model_dict['model_state'])
        
        else: 
            dmm = DeepMarkovModel(
                input_dim=self._input_dim, 
                z_dim=self._z_dim, 
                emission_dim=self._emission_dim,
                transition_dim=self._transition_dim, 
                rnn_dim=self._rnn_dim, 
                rnn_dropout_rate=self._rnn_dropout_rate,
                variance=self._variance
            )

        return dmm
    
    def __initOptimizer(self):
        return pyro.optim.Adam({"lr": 1e-3})
    
    def __initStepModel(self):
        return self.dmm.model
    
    def __initGuide(self):
        return self.dmm.guide

    def __initTrainDl(self, x_train, batch_size, num_workers, sequence_length):
        train_dl = StockDataset(x_train, sequence_length=sequence_length)

        train_dl = DataLoader(train_dl, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    # pin_memory=self.use_cuda,
                                    shuffle=False
                                    )

        self.__batch_size = batch_size
        self.__num_workers = num_workers
        self.__sequence_length = sequence_length

        return train_dl

    def __initValDl(self, x_test):
        val_dl = StockDataset(x_test, 
                                sequence_length=self.__sequence_length
                                )

        val_dl = DataLoader(val_dl, 
                                    batch_size=self.__batch_size, 
                                    num_workers=self.__num_workers,
                                    # pin_memory=self.use_cuda,
                                    shuffle=False
                                    )
        
        return val_dl

    def fit(self, 
            x_train,
            epochs=10,
            sequence_length=30,
            batch_size=8, 
            num_workers=4,
            validation_sequence=30, 
            ):
        
        scaler = normalize(x_train)

        x_train = scaler.fit_transform()
        val_dl = x_train[-validation_sequence:]
        x_train = x_train[:len(x_train)-len(val_dl)]

        train_dl = self.__initTrainDl(x_train, 
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        sequence_length=sequence_length
                                        )

        val_dl = self.__initValDl(val_dl)

        self.svi = SVI(self.step_model, 
                  self.step_guide, 
                  self.optimizer, 
                  loss=Trace_ELBO()
                )
        
        validation_cadence = 5

        for epoch_ndx in tqdm((range(1, epochs + 1)),position=0, leave=True):
            epoch_loss = 0.0
            for x_batch, y_batch in train_dl:  
                loss  = self.svi.step(
                    x_data = x_batch,
                    y_data = y_batch
                )
                epoch_loss += loss
                
            print(f"Epoch {epoch_ndx}, Loss: {epoch_loss / len(train_dl)}")
                
            if epoch_ndx % validation_cadence == 0:

                total_loss = 0 
                self.dmm.rnn.eval()   # Turns off training-time behaviour
                    
                for x_batch, y_batch in val_dl:
                        loss_var = self.svi.evaluate_loss(x_batch, y_batch) 
                        total_loss += loss_var / val_dl.batch_size
                print(f"Epoch {epoch_ndx}, Val Loss {total_loss}")
    
    def predict(self, x_test):
        # set the model to evaluation mode
        self.dmm.eval()

        # transform x_test in data loader
        scaler = normalize(x_test)
        x_test = scaler.fit_transform()
        self.std = scaler.std()
        self.mean = scaler.mean()

        val_dl = self.__initValDl(x_test)

        # create a list to hold the predicted y values
        predicted_y = []

        # iterate over the test data in batches
        for x_batch, y_batch in val_dl:
            # make predictions for the current batch
            with torch.no_grad():
                # compute the mean of the emission distribution for each time step
                *_, z_loc, z_scale = self.dmm.guide(x_batch)
                z_scale = F.softplus(z_scale)
                z_t = dist.Normal(z_loc, z_scale).rsample()
                mean_t, _ = self.dmm.emitter(z_t, x_batch)
                
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


    def sample(self, num_samples):
        # set the model to evaluation mode
        self.dmm.eval()

        # generate the latent z's
        with torch.no_grad():
            # initialize z_prev to the prior distribution
            z_prev = dist.Normal(self.dmm.z_0, 1).rsample((num_samples,))

            # initialize the list of sampled z's
            z_list = [z_prev]

            # iterate over the time steps
            for t in range(1, self.__sequence_length + 1):
                # sample the next z from the prior distribution
                z_loc, z_scale = self.dmm.transition(z_prev, 
                                                     torch.zeros(num_samples, 
                                                                 self._input_dim)
                                                                 )
                z_t = dist.Normal(z_loc, z_scale).rsample()
                z_list.append(z_t)

                # set z_prev to the sampled z for the next time step
                z_prev = z_t

        # generate the observations y's
        with torch.no_grad():
            # initialize the list of sampled y's
            y_list = []

            # iterate over the time steps
            for t in range(1, self.__sequence_length + 1):
                # sample the y from the emission distribution
                y_t, _ = self.dmm.emitter(z_list[t], torch.zeros(num_samples, self._input_dim))
                y_list.append(y_t * self.std + self.mean)

        # concatenate the sampled y's into a single tensor
        y_samples = torch.stack(y_list, dim=1)

        # return the generated samples as a numpy array of shape [num_samples, 1]
        return y_samples[:, -1].squeeze().numpy().reshape(-1,1)