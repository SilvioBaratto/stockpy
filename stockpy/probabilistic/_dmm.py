# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
An implementation of a Deep Markov Model in Pyro based on reference [1].
This is essentially the DKS variant outlined in the paper. The primary difference
between this implementation and theirs is that in our version any KL divergence terms
in the ELBO are estimated via sampling, while they make use of the analytic formulae.
We also illustrate the use of normalizing flows in the variational distribution (in which
case analytic formulae for the KL divergences are in any case unavailable).
Reference:
[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""
import datetime
import hashlib
import os
import shutil
import sys
import glob
sys.path.append("..")
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
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
)
from pyro.optim import ClippedAdam

from util.StockDataset import StockDatasetSequence, normalize

from util.logconf import logging
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
from sklearn.preprocessing import StandardScaler

# set style of graphs
plt.style.use('ggplot')
from pylab import rcParams
plt.rcParams['figure.dpi'] = 100


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)
from util.StockDataset import StockDataset, normalize

# TODO Implement forecasting function and plotting
# TODO Implement interface 


class Emitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        return ps


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
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


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, lstm_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, lstm_dim)
        self.lin_hidden_to_loc = nn.Linear(lstm_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(lstm_dim, z_dim)
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


class DeepMarkovModel(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(
        self,
        input_dim=4,
        z_dim=16,
        emission_dim=16,
        transition_dim=32,
        lstm_dim=64,
        num_layers=1,
        num_iafs=0,
        iaf_dim=64,
        use_cuda=False,
    ):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, lstm_dim)
        # dropout just takes effect on inner layers of rnn
        self.num_layers = num_layers
        self.hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_dim, lstm_dim, num_layers, batch_first=True)

        # if we're using normalizing flows, instantiate those too
        self.iafs = [
            affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)
        ]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, lstm_dim))

    def model(self, x):

        # this is the number of time steps we need to process in the mini-batch
        T_max = x.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(x.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(x)):
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
                z_t = pyro.sample("z_%d" % t,
                                    dist.Normal(z_loc, z_scale)
                                        .mask(x[:, t - 1:t])
                                        .to_event(1))

                # compute the probabilities that parameterize the bernoulli likelihood
                emission_probs_t = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # bernoulli distribution p(x_t|z_t)
                pyro.sample("obs_x_%d" % t,
                            dist.Bernoulli(emission_probs_t)
                                .mask(x[:, t - 1:t])
                                .to_event(1),
                            obs=x[:, t - 1, :])
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(self, x):

        # this is the number of time steps we need to process in the mini-batch
        T_max = x.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
        _, (rnn_output, _) = self.lstm(x, (h0, c0))
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        # reverse the time-ordering in the hidden state and un-pack it
        # rnn_output = poly.pad_and_reverse(rnn_output, T_max)
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(x.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(x)):
            # sample the latents z one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(
                        dist.Normal(z_loc, z_scale), self.iafs
                    )
                    assert z_dist.event_shape == (self.z_q_0.size(0),)
                    assert z_dist.batch_shape[-1:] == (len(x),)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                    assert z_dist.event_shape == ()
                    assert z_dist.batch_shape[-2:] == (
                        len(x),
                        self.z_q_0.size(0),
                    )
                print(type(z_dist))
                # sample z_t from the distribution z_dist
                if len(self.iafs) > 0:
                    # in output of normalizing flow, all dimensions are correlated (event shape is not empty)
                    # z_t = pyro.sample(
                    #    "z_%d" % t, z_dist.mask(x[:, t - 1])
                    #    )
                    z_t = z_dist.mask(x[:, t - 1])
                else:
                    # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                    # z_t = pyro.sample(
                    #        "z_%d" % t,
                    #        z_dist.mask(x[:, t - 1 : t]).to_event(1),
                    #)
                    print(z_dist.mask(x[:, t - 1 : t]).to_event(1))
                    z_t = z_dist.mask(x[:, t - 1 : t]).to_event(1),
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t

class DMM():

    def __init__(self,
                input_size=4,
                hidden_size=32, 
                num_layers=1,
                dropout=0.1,
                pretrained=False
                ):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.pretrained = pretrained
        
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        
        # self.model_path = self.__initModelPath()
        self.model = self.__initModel()
        self.optimizer = self.__initOptimizer()
        self.inference = self.__initInference()

    def __initModelPath(self):
        local_path = os.path.join(
                '..',
                '..',
                'models',
                'DMM',
                'DMM{}.state'.format('*', 'best'),
        )

        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(
                    '..',
                    '..',
                    'models',
                    'DMM',
                    'DMM{}.state'.format('*'),
                )
            file_list = glob.glob(pretrained_path)

        else:
            pretrained_path = None

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            raise

    def __initModel(self):

        if self.pretrained:
            model_dict = torch.load(self.model_path)

            model = DeepMarkovModel()

            model.load_state_dict(model_dict['model_state'])
        
        else: 
            model = DeepMarkovModel()

        return model

    def __initOptimizer(self):
        # setup optimizer
        adam_params = {
            "lr": 0.0003,
            "betas": (0.96, 0.999),
            "clip_norm": 10.0,
            "lrd": 0.99996,
            "weight_decay": 2.0,
        }
        adam = ClippedAdam(adam_params)
        return adam

    def __initInference(self):
        elbo = Trace_ELBO()
        svi = SVI(self.model.model, 
                    self.model.guide, 
                    self.optimizer, 
                    loss=elbo)
        return svi

    def __initTrainDl(self, x_train, batch_size, num_workers, sequence_length):
        train_dl = StockDatasetSequence(x_train, sequence_length=sequence_length)

        train_dl = DataLoader(train_dl, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    # pin_memory=self.use_cuda,
                                    shuffle=True
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

    def computeBatchLoss(self, x):
        loss = self.inference.step(x)
        # keep track of the training loss
        return loss

    def fit(self, 
            x_train, 
            epochs=10, 
            batch_size=8, 
            num_workers=4,
            sequence_length=30,
            save_model=True,
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

        best_score = 0.0
        total_loss = 0
        validation_cadence = 5
        
        for epoch_ndx in tqdm((range(1, epochs + 1)),position=0, leave=True):
            self.model.train()

            # batch_iter = enumerate(train_dl)

            for x, y in train_dl:
                # self.optimizer.zero_grad()  # Frees any leftover gradient tensors

                loss_var = self.computeBatchLoss(x)

                # loss_var.backward()     # Actually updates the model weights
                # self.optimizer.step()
                total_loss += loss_var
        
            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                loss_val = self.doValidation(epoch_ndx, val_dl)
                best_score = max(loss_val, best_score)
                # self.saveModel('LSTM', epoch_ndx, loss_val == best_score)

    def doValidation(self, 
                    epoch_ndx, 
                    val_dl):
        total_loss = 0
        with torch.no_grad():
            self.model.eval()   # Turns off training-time behaviour

            batch_iter = enumerate(val_dl)

            for batch_ndx, batch_tup in batch_iter:
                loss_var = self.computeBatchLoss(batch_tup)
                total_loss += loss_var

        return total_loss / len(val_dl)

    def predict(self, 
                x_test,   
                plot=False
                ):

        scaler = normalize(x_test)
        x_test = scaler.fit_transform()
        val_dl = self.__initValDl(x_test)
        batch_iter = enumerate(val_dl)

        output = torch.tensor([])
        self.model.eval()
        with torch.no_grad():
            for batch_ndx, batch_tup in batch_iter:
                y_star = self.model(batch_tup[0])
                output = torch.cat((output, y_star), 0)
        
        if plot is True:
            y_pred = output * scaler.std() + scaler.mean() # * self.std_test + self.mean_test 
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
            
        return output * scaler.std() + scaler.mean()# * self.std_test + self.mean_test 

    def forecast(forecast, look_back, plot=False):
        pass

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            '..',
            '..',
            'models',
            'DMM',
            '{}_{}_{}.state'.format(
                    type_str,
                    self.hidden_dim,
                    self.num_layers
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx
        }
        torch.save(state, file_path)

        # log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                '..',
                '..',
                'models',
                'DMM',
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.hidden_dim,
                    self.num_layers,
                    'best',
                )
            )
            shutil.copyfile(file_path, best_path)

            # log.debug("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            hashlib.sha1(f.read()).hexdigest()

    def initModelPath(self, type_str):
        local_path = os.path.join(
            '..',
            '..',
            'models',
            type_str + '_{}.state'.format('*', '*', 'best'),
        )

        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(
                '..',
                '..',
                'models',
                type_str + '_{}_{}.{}.state'.format('*', '*', '*'),
            )
            file_list = glob.glob(pretrained_path)
        else:
            pretrained_path = None

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([local_path, pretrained_path, file_list])
            raise