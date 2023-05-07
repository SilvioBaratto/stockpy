import os
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import pyro.poutine as poutine
from typing import Tuple, Optional
from ._base_model import BaseRegressorRNN
from ._base_model import BaseClassifierRNN
from ._hmm_utils import Emitter, GatedTransition
from ..config import Config as cfg

class GaussianHMMRegressor(BaseRegressorRNN):
    """
    This PyTorch Module encapsulates the model as well as the variational distribution (the guide)
    for the Gaussian Hidden Markov Model.

    :param input_size: the number of input features
    :type input_size: int
    :param z_dim: The dimensionality of the latent variables z.
    :type z_dim: int
    :param emission_dim: The dimensionality of the hidden state in the Emitter network.
    :type emission_dim: int
    :param transition_dim: The dimensionality of the hidden state in the Gated Transition network.
    :type transition_dim: int
    :param variance: The initial variance value for the observation distribution.
    :type variance: float

    :ivar emitter: the Emitter module which parameterizes the Gaussian observation likelihood p(y_t | z_t, x_t)
    :vartype emitter: Emitter
    :ivar transition: the GatedTransition module which parameterizes the dynamics of the latent variables z_t
    :vartype transition: GatedTransition
    :ivar z_0: a trainable parameter representing the initial latent state
    :vartype z_0: torch.nn.Parameter
    :ivar z_q_0: a trainable parameter representing the initial latent state for the variational distribution
    :vartype z_q_0: torch.nn.Parameter
    :example:
        >>> from stockpy.probabilistic import GaussianHMM
        >>> gaussian_hmm = GaussianHMM()
    """
    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(input_size=input_size, output_size=output_size)
        self.transition = GatedTransition(input_size=input_size, output_size=output_size)

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
        This function defines the generative process of the Gaussian Hidden Markov Model (GaussianHMM).
        It takes a mini-batch of input sequences `x_data`, and if available, their corresponding
        output sequences `y_data`, and generates a sequence of latent variables `z_t` at each time step.
        It uses a DNN to model the emission probability distribution `p(y_t|z_t, x_t)` and 
        another DNN to model the transition probability distribution `p(z_t|z_{t-1}, x_t)`.
        It returns the final sampled latent variable `z_t`.

        :param x_data: input data of size (batch_size, num_time_steps, input_size)
        :type x_data: torch.Tensor
        :param y_data: output data of size (batch_size, num_time_steps, output_size), defaults to None
        :type y_data: Optional[torch.Tensor], optional
        :param annealing_factor: controls the weight of the KL divergence term, defaults to 1.0
        :type annealing_factor: float, optional

        :returns: tensor of size (batch_size, latent_dim) representing the final sampled latent variable
        :rtype: torch.Tensor
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
        Defines the guide function for the Deep Markov Model. This function is used
        for the variational inference procedure in Pyro.

        :param x_data: The input data tensor of shape (batch_size, T_max, input_size)
        :type x_data: torch.Tensor
        :param y_data: The output data tensor of shape (batch_size, T_max, output_size), defaults to None
        :type y_data: Optional[torch.Tensor], optional
        :param annealing_factor: The scaling factor for the KL term in the loss function, defaults to 1.0
        :type annealing_factor: float, optional

        :returns: The latent variable tensor sampled at the final time step, of shape (batch_size, z_dim)
        :rtype: torch.Tensor
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