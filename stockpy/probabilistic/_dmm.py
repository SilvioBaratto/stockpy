import os
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import pyro.poutine as poutine
import torch.nn.functional as F
from typing import Tuple, Optional
from ._base_model import BaseRegressorRNN
from ._base_model import BaseClassifierRNN
from ._hmm_utils import Emitter, GatedTransition, Combiner
from ..config import Config as cfg

class DeepMarkovModelRegressor(BaseRegressorRNN):
    """
    The DeepMarkovModel class implements a Deep Markov Model, which is a
    generative model for time series data that combines recurrent neural networks
    (RNNs) and hidden Markov models (HMMs). It consists of an RNN for encoding
    input sequences, a Combiner module to combine the RNN hidden states with the
    previous latent states, and an Emitter and GatedTransition module to model
    the emission and transition probability distributions.

    :param input_size: the number of input features
    :type input_size: int
    :param rnn_dim: The number of hidden units in the RNN used for encoding the input sequences.
    :type rnn_dim: int
    :param z_dim: The dimensionality of the latent variables z.
    :type z_dim: int
    :param emission_dim: The dimensionality of the hidden state in the Emitter network.
    :type emission_dim: int
    :param transition_dim: The dimensionality of the hidden state in the Gated Transition network.
    :type transition_dim: int
    :param variance: The initial variance value for the observation distribution.
    :type variance: float

    :ivar emitter: Emitter module for modeling the emission probability distribution
    :vartype emitter: Emitter
    :ivar transition: GatedTransition module for modeling the transition probability distribution
    :vartype transition: GatedTransition
    :ivar combiner: Combiner module for combining RNN hidden states with previous latent states
    :vartype combiner: Combiner
    :ivar rnn: GRU-based RNN for encoding input sequences
    :vartype rnn: nn.GRU
    :ivar z_0: Initial latent state parameter
    :vartype z_0: nn.Parameter
    :ivar z_q_0: Initial variational latent state parameter
    :vartype z_q_0: nn.Parameter
    :ivar h_0: Initial RNN hidden state parameter
    :vartype h_0: nn.Parameter

    :example:
        >>> from stockpy.probabilistic import DeepMarkovModel
        >>> deep_markov_model = DeepMarkovModel()
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
        self.combiner = Combiner(input_size=input_size, output_size=output_size)

        self.rnn = nn.GRU(input_size=input_size, 
                          hidden_size=cfg.prob.rnn_dim,
                          batch_first=True,
                          bidirectional=False, 
                          num_layers=2
                          )

        if cfg.training.use_cuda:
            if torch.cuda.device_count() > 1:
                self.emitter = nn.DataParallel(self.emitter)
                self.transition = nn.DataParallel(self.transition)
                self.combiner = nn.DataParallel(self.combiner)
                self.rnn = nn.DataParallel(self.rnn)

            self.emitter = self.emitter.to(cfg.training.device)
            self.transition = self.transition.to(cfg.training.device)
            self.combiner = self.combiner.to(cfg.training.device)
            self.rnn = self.rnn.to(cfg.training.device)

        # define a (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(cfg.prob.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(1, cfg.prob.z_dim))
        # define a (trainable) parameter for the initial hidden state of the RNN
        self.h_0 = nn.Parameter(torch.zeros(1, 1, cfg.prob.rnn_dim))

    def model(self, 
              x_data: torch.Tensor, 
              y_data: Optional[torch.Tensor] = None, 
              annealing_factor: float = 1.0
              ) -> torch.Tensor:
        """
        This function defines the generative process of the Deep Markov Model (DMM). 
        It takes a mini-batch of input sequences `x_data` and, if available, their corresponding
        output sequences `y_data`, and generates a sequence of latent variables `z_t` at each time step.
        It uses an RNN to encode the input sequences, and a Combiner module to combine the RNN hidden states
        with the previous latent states. The Emitter and GatedTransition modules are then used to model the
        emission and transition probability distributions, respectively. It returns the final sampled latent
        variable `z_t`.
        
        :param x_data: input data of size (batch_size, num_time_steps, input_size)
        :type x_data: torch.Tensor
        :param y_data: output data of size (batch_size, num_time_steps, output_size), defaults to None
        :type y_data: Optional[torch.Tensor], optional
        :param annealing_factor: controls the weight of the KL divergence term, defaults to 1.0
        :type annealing_factor: float, optional

        :returns: tensor of size (batch_size, latent_dim) representing the final sampled latent variable
        :rtype: torch.Tensor
        
        """

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
            
    def guide(self,
              x_data: torch.Tensor, 
              y_data: Optional[torch.Tensor] = None, 
              annealing_factor: float = 1.0
              ) -> torch.Tensor:
        """
        This function defines the variational approximation (inference network) of the Deep Markov Model (DMM). 
        It takes a mini-batch of input sequences `x_data` and, if available, their corresponding
        output sequences `y_data`, and generates a sequence of latent variables `z_t` at each time step.
        The guide function aims to approximate the true posterior distribution over the latent variables
        given the input sequences. It uses an RNN to encode the input sequences, and a Combiner module to 
        combine the RNN hidden states with the previous latent states. It returns the final sampled latent
        variable `z_t`.

        :param x_data: The input data tensor of shape (batch_size, T_max, input_size)
        :type x_data: torch.Tensor
        :param y_data: The output data tensor of shape (batch_size, T_max, output_size), defaults to None
        :type y_data: Optional[torch.Tensor], optional
        :param annealing_factor: The scaling factor for the KL term in the loss function, defaults to 1.0
        :type annealing_factor: float, optional

        :returns: The latent variable tensor sampled at the final time step, of shape (batch_size, z_dim)
        :rtype: torch.Tensor
        """
        
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
        
        z_q_0_expanded = self.z_q_0.expand(x_data.size(0), cfg.prob.z_dim)
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
           
                # the latent sampled at this time step will be conditioned
                # upon in the next time step so keep track of it
                z_prev = z_t

            return z_t