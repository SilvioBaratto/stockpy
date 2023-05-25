from abc import abstractmethod, ABCMeta
import os
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoNormal
import torch.nn.functional as F
import pyro.poutine as poutine
from pyro.infer import (
    SVI,
    Trace_ELBO,
    Predictive,
    TraceMeanField_ELBO
)
from typing import Union, Tuple, Optional
import pandas as pd
import numpy as np
from ._base import RegressorProb
from .utils._hmm_utils import Emitter, GatedTransition
from ..config import Config as cfg

class GHMMRegressor(RegressorProb):
    """
    A class used to represent a Gaussian Hidden Markov Model (GHMM) for regression tasks.
    This class inherits from the `RegressorProb` class.

    Attributes:
        rnn_dim (int): 
            The dimension of the hidden state of the RNN.
        z_dim (int): 
            The dimension of the latent random variable z.
        emission_dim (int): 
            The dimension of the hidden state of the emission model.
        transition_dim (int): 
            The dimension of the hidden state of the transition model.
        variance (float): 
            The variance of the observation noise.
        model_type (str): 
            A string representing the type of the model (default is "rnn").

    Methods:
        __init__(self, **kwargs): 
            Initializes the GaussianHMMRegressor object with given or default parameters.
        _init_model(self): 
            Initializes the GHMM modules (emitter, transition) and some trainable parameters.
        model(self, x_data: torch.Tensor, y_data: Optional[torch.Tensor] = None, annealing_factor: float = 1.0) -> torch.Tensor:
            Defines the generative model which describes the process of generating the data.
        guide(self, x_data: torch.Tensor, y_data: Optional[torch.Tensor] = None, annealing_factor: float = 1.0) -> torch.Tensor:
            Defines the variational guide (approximate posterior) that is used for inference.
    """

    model_type = "rnn"
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        # Initialize GHMM modules: emitter, transition
        # Also define trainable parameters z_0, z_q_0, and h_0 that help define the 
        # probability distributions p(z_1) and q(z_1)

        self.emitter = Emitter(input_size=self.input_size, 
                               output_size=self.output_size)
        self.transition = GatedTransition(input_size=self.input_size, 
                                          output_size=self.output_size)

        if cfg.training.use_cuda:
            if torch.cuda.device_count() > 1:
                self.emitter = nn.DataParallel(self.emitter)
                self.transition = nn.DataParallel(self.transition)

            self.emitter = self.emitter.to(cfg.training.device)
            self.transition = self.transition.to(cfg.training.device)

        # define a (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(cfg.prob.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(1, cfg.prob.z_dim))

    def model(self, 
              x_data: torch.Tensor, 
              y_data: Optional[torch.Tensor] = None, 
              annealing_factor: float = 1.0
              ) -> torch.Tensor:
        """
        Implements the generative model p(y,z|x) which includes the observation 
        model p(y|z) and transition model p(z_t | z_{t-1}). It also handles the 
        computation of the parameters of these models.

        Args:
            x_data (torch.Tensor): Input tensor for the model.
            y_data (Optional[torch.Tensor]): Optional observed output tensor for the model.
            annealing_factor (float, optional): Annealing factor used in poutine.scale to handle KL annealing.

        Returns:
            torch.Tensor: The sampled latent variable `z` from the last time step of the model.
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

    def guide(self, 
              x_data: torch.Tensor, 
              y_data: Optional[torch.Tensor] = None, 
              annealing_factor: float = 1.0
              ) -> torch.Tensor:
        
        """
        Implements the guide (also called the inference model or variational distribution) q(z|x,y)
        which is an approximation to the posterior p(z|x,y). It also handles the computation of the 
        parameters of this guide.

        Args:
            x_data (torch.Tensor): Input tensor for the guide.
            y_data (Optional[torch.Tensor]): Optional observed output tensor for the guide.
            annealing_factor (float, optional): Annealing factor used in poutine.scale to handle KL annealing.

        Returns:
            torch.Tensor: The sampled latent variable `z` from the last time step of the guide.
        """
        
        T_max = x_data.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("ghmm", self)

        # initialize the values of the latent variables using heuristics
        z_q_0_expanded = self.z_q_0.expand(x_data.size(0), cfg.prob.z_dim)
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
        
    def _initSVI(self) -> pyro.infer.svi.SVI:
        """
        Initializes the Stochastic Variational Inference (SVI) for the GaussianHMMRegressor model.

        Returns:
            pyro.infer.svi.SVI: The initialized SVI object.
        """
        return SVI(model=self.model,
                   guide=self.guide,
                   optim=self.optimizer, 
                   loss=TraceMeanField_ELBO())

    def _predict(self,
                test_dl : torch.utils.data.DataLoader
                ) -> torch.Tensor:

        return self._predictHMM(test_dl)
    
    def forward(self):
        pass