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
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    config_enumerate,
    Predictive
)
from pyro.optim import ClippedAdam

from util.StockDataset import StockDatasetSequence, normalize
import pandas as pd
import matplotlib.pyplot as plt

class Emitter(PyroModule):
    """
    Parameterizes the normal observation likelihood p(x_t | z_t)
    """
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 emission_dim
                 ):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_mu = nn.Linear(emission_dim, input_dim)
        self.lin_hidden_to_var = nn.Linear(emission_dim, input_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the mean and
        variance of the normal distribution p(x_t|z_t)
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        mu = self.lin_hidden_to_mu(h2)
        var = self.lin_hidden_to_var(h2).exp()
        return mu, var
    

class GatedTransition(PyroModule):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, 
                 z_dim=32, 
                 transition_dim=64,
                 ):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale
    
class Combiner(PyroModule):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, 
                 z_dim=32, 
                 rnn_dim=32
                 ):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale
    
class DeepMarkovModel(PyroModule):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(
        self,
        input_dim=4,
        z_dim=32,
        emission_dim=32,
        transition_dim=64,
        rnn_dim=32,
        num_layers=1,
        output_dim=1
    ):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.combiner = Combiner(z_dim, rnn_dim)
        # dropout just takes effect on inner layers of rnn
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=rnn_dim,
            batch_first=True,
            num_layers=num_layers,
        )
        self.input_dim = input_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.z_dim = z_dim
        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

    def model(self, 
              x_data, 
              y_data=None, 
              annealing_factor=1.0
              ):
        
        self.emitter = Emitter(x_data.size(0), self.z_dim, self.emission_dim)
        self.trans = GatedTransition(self.z_dim, self.transition_dim)
        # this is the number of time steps we need to process in the mini-batch
        T_max = x_data.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(x_data.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", x_data.shape[0]):
            # sample the latents z and observed x's one time step at a time
            for t in range(1, T_max + 1):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing. we use the mask() method to deal with raggedness
                # in the observed data (i.e. different sequences in the mini-batch
                # have different lengths)

                # first compute the parameters of the diagonal gaussian
                # distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # then sample z_t according to dist.Normal(z_loc, z_scale).
                # note that we use the reshape method so that the univariate
                # Normal distribution is treated as a multivariate Normal
                # distribution with a diagonal covariance.
                
                with poutine.scale(None, annealing_factor):
                    z_t = pyro.sample("z_%d" % t,
                                    dist.Normal(z_loc, z_scale)
                                                .to_event(1))

                # compute the probabilities that parameterize the Normal likelihood
                # emission_probs_t = self.emitter(z_t)
                mu, var = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # Normal distribution p(x_t|z_t)
                pyro.sample("obs_x_%d" % t,
                            dist.Normal(mu, var).to_event(1),
                                         obs=y_data)                                   
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

    def guide(self, 
              x_data, 
              y_data=None, 
              annealing_factor=1.0
              ):

        # this is the number of time steps we need to process in the mini-batch
        T_max = x_data.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(1, x_data.size(0),
                                    self.rnn.hidden_size).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(x_data, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(x_data.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", x_data.shape[0]):
            # sample the latents z one time step at a time
            for t in range(1, T_max + 1):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                # z_dist = dist.Normal(z_loc, z_scale)
                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(None, annealing_factor):
                    z_t = pyro.sample("z_%d" % t, 
                                      dist.Normal(z_loc, z_scale)
                                        .to_event(1))
                # the latent sampled at this time step will be conditioned
                # upon in the next time step so keep track of it
                z_prev = z_t

        return z_t
    
class DMM(PyroModule):

    def __init__(self, 
                pretrained=False
                ):
        # initialize PyroModule
        super(DMM, self).__init__()

        self.pretrained = pretrained
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        # self.model_path = self.__initModelPath()
        self.dmm = self.__initModel()
        self.optimizer = self.__initOptimizer()
        self.step_model = self.__initStepModel()
        self.step_guide = self.__initGuide()

        self.name = "bayesian_network"

    def __initModel(self):

        if self.pretrained:
            model_dict = torch.load(self.model_path)

            dmm = DeepMarkovModel()

            dmm.load_state_dict(model_dict['model_state'])
        
        else: 
            dmm = DeepMarkovModel()

        return dmm
    
    def __initOptimizer(self):
        adam_params = {
            "lr": 0.0003,
            "betas": (0.96, 0.999),
            "clip_norm": 10.0,
            "lrd": 0.99996,
            "weight_decay": 2.0,
        }
        return ClippedAdam(adam_params)
    
    def __initStepModel(self):
        return self.dmm.model
    
    def __initGuide(self):
        return self.dmm.guide

    def __initTrainDl(self, x_train, batch_size, num_workers, sequence_length):
        train_dl = StockDatasetSequence(x_train, sequence_length=sequence_length)

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
        val_dl = StockDatasetSequence(x_test, 
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

        pyro.clear_param_store()
        for epoch_ndx in tqdm((range(1, epochs + 1)),position=0, leave=True):
            loss = 0.0
            self.dmm.rnn.train()
            for x_batch, y_batch in train_dl:        
                loss = self.svi.step(x_data=x_batch, y_data=y_batch)
 

    def doValidation(self, val_dl):
        total_loss = 0
        self.dmm.rnn.eval()   # Turns off training-time behaviour
            
        for x_batch, y_batch in val_dl:
            loss_var = self.svi.evaluate_loss(x_batch, y_batch) 
            total_loss += loss_var

        return total_loss / len(val_dl)

    def predict(self, 
                x_test, 
                plot=False
                ):

        scaler = normalize(x_test)
        x_test = scaler.fit_transform()
        test_loader = self.__initValDl(x_test)

        output = torch.tensor([])
        for x_batch, y_batch in test_loader:
            predictive = Predictive(self.step_model, 
                                    guide=self.step_guide, 
                                    num_samples=self.__batch_size,
                                    return_sites=("_RETURN")
                                    )

            y_pred = predictive(x_batch)
            # samples = self.summary(y_pred)

        if plot is True:
            y_pred = output.detach().numpy() * scaler.std() + scaler.mean() # * self.std_test + self.mean_test 
            y_test = (x_test['Close']).values * scaler.std() + scaler.mean() # * self.std_test + self.mean_test
            test_data = x_test[0: len(x_test)]
            days = np.array(test_data.index, dtype="datetime64[ms]")
            
            fig = plt.figure()
            
            axes = fig.add_subplot(111)
            axes.plot(days, y_test, 'bo-', label="actual") 
            axes.plot(days, y_pred, 'r+-', label="predicted")
            
            fig.autofmt_xdate()
            
            plt.legend()
            plt.show()
        
        return y_pred, scaler.std(), scaler.mean() # .detach().numpy() * scaler.std() + scaler.mean()

    def forward(self, x_batch, n_samples=10):
        """ Compute predictions on `inputs`. 
        `n_samples` is the number of samples from the posterior distribution.
        If `sample_idx` is provided, it is used as a seed for sampling a single
        model from the Variational family.
        If `avg_prediction` is True, it returns the average prediction on 
        `inputs`, otherwise it returns all predictions 
        """
        preds = []
        # take multiple samples
        for _ in range(n_samples):         
            guide_trace = poutine.trace(self.step_guide).get_trace(x_batch)
            preds.append(guide_trace.nodes['_RETURN']['value'])
        
        # list of tensors to tensor
        # preds.shape = (n_samples, batch_size, n_classes)
        preds = torch.stack(preds)

        # return predictions 
        return preds
    
    def _predict(self, x_test):
        scaler = normalize(x_test)
        x_test = scaler.fit_transform()
        test_loader = self.__initValDl(x_test)

        output = torch.tensor([])
        for x_batch, y_batch in test_loader:

            samples = self.forward(x_batch, n_samples=self.__batch_size)

            y_pred = torch.mean(samples, 0)
            output = torch.cat((output, y_pred), 0)
        
        return output.detach().numpy() * scaler.std() + scaler.mean()

    @staticmethod
    def summary(samples):
        site_stats = {}
        for k, v in samples.items():
            site_stats[k] = {
                "mean": torch.mean(v, 0)
            }
        return site_stats