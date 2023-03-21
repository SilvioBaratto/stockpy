import sys
import datetime
import hashlib
import os
import shutil
import sys
import glob
from os.path import exists
sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import pyro.poutine as poutine
from pyro.infer import (
    SVI,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceMeanField_ELBO
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
        self.dmm = self._initModel()
        self.optimizer = self._initOptimizer()
        self.step_model = self._initStepModel()
        self.step_guide = self._initGuide()

        self.name = "deep_markov_network"

    def _initModel(self):

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
    
    def _initOptimizer(self):
        return pyro.optim.Adam({"lr": 1e-3})
    
    def _initStepModel(self):
        return self.dmm.model
    
    def _initGuide(self):
        return self.dmm.guide

    def _initTrainDl(self, x_train, batch_size, num_workers, sequence_length):
        train_dl = StockDataset(x_train, sequence_length=sequence_length)

        train_dl = DataLoader(train_dl, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    # pin_memory=self.use_cuda,
                                    shuffle=False
                                    )

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._sequence_length = sequence_length

        return train_dl

    def _initValDl(self, x_test):
        val_dl = StockDataset(x_test, 
                                sequence_length=self._sequence_length
                                )

        val_dl = DataLoader(val_dl, 
                                    batch_size=self._batch_size, 
                                    num_workers=self._num_workers,
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
            patience=5
            ):
        
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

        self.svi = SVI(self.step_model, 
                  self.step_guide, 
                  self.optimizer, 
                  loss=TraceMeanField_ELBO()
                )
        
        validation_cadence = 5
        best_loss = float('inf')
        counter = 0

        for epoch_ndx in tqdm((range(1, epochs + 1)), position=0, leave=True):
            epoch_loss = 0.0
            for x_batch, y_batch in train_dl:  
                epoch_loss += self._computeBatchLoss(x_batch, y_batch)

            if epoch_ndx % validation_cadence != 0:                
                print(f"Epoch {epoch_ndx}, Loss: {epoch_loss / len(train_dl)}")

            else:
                total_loss = self._doValidation(val_dl)

                print(f"Epoch {epoch_ndx}, Val Loss {total_loss}")

                if total_loss < best_loss:
                    best_loss = total_loss
                    counter = 0
                else:
                    counter += 1

                if counter >= patience:
                    print(f"No improvement after {patience} epochs. Stopping early.")
                    break

                self.dmm.rnn.train()

    def _computeBatchLoss(self,
                          x_batch,
                          y_batch
                          ):

        loss = self.svi.step(
            x_data=x_batch,
            y_data=y_batch
        )
        # keep track of the training loss
        return loss  # This is the loss over the entire batch
    
    def _doValidation(self, val_dl):
        total_loss = 0
        self.dmm.rnn.eval()
        
        for x_batch, y_batch in val_dl:
            loss_var = self.svi.evaluate_loss(x_batch, y_batch) 
            total_loss += loss_var / val_dl.batch_size

        return total_loss 
    
    def predict(self, x_test):
        # set the model to evaluation mode
        self.dmm.eval()

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
    
    def trading(self, 
                x_data, 
                shares=0, 
                stop_loss=0.0,
                initial_balance=10000, 
                plot=True
                ):
        
        y_pred = self.predict(x_test=x_data)

        # compute the buy/sell signals
        buy_signals = (y_pred[1:] > y_pred[:-1]).astype(int)
        sell_signals = (y_pred[1:] < y_pred[:-1]).astype(int)

        y_test = x_data['Close'].values

        # simulate the trades
        balance = initial_balance
        for i in range(len(y_test) - 1):
            if buy_signals[i] == 1:
                price = y_test[i + 1] * 100  # price is the last feature value scaled by 100
                num_shares = int(balance / price)
                shares += num_shares
                balance -= num_shares * price
            elif sell_signals[i] == 1:
                price = y_test[i + 1] * 100  # price is the last feature value scaled by 100
                balance += shares * price
                shares = 0
            # implement stop-loss strategy
            elif y_test[i + 1] < stop_loss * y_test[0]:
                balance += shares * y_test[i + 1] * 100
                shares = 0

        # compute the final balance
        final_balance = balance + (shares * y_test[-1] * 100)

        # plot the predicted values and the buy/sell signals
        y_pred = y_pred[1:]

        if plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(y_pred)
            ax.scatter(np.where(buy_signals == 1)[0], y_pred[buy_signals == 1], 
                       c='g', marker='^', s=100, label='Buy')
            ax.scatter(np.where(sell_signals == 1)[0], y_pred[sell_signals == 1], 
                       c='r', marker='v', s=100, label='Sell')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.set_title('Trading Simulation')
            fig.autofmt_xdate()
            ax.legend()
            ax.text(0.05, 0.05, 
                    f'Final balance: ${initial_balance / final_balance * 100:.2f}%', 
                    ha='left', va='center',
                      transform=ax.transAxes, 
                      bbox=dict(facecolor='white', alpha=0.5)
                      )

        return initial_balance / final_balance * 100
    
    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            '..',
            '..',
            'models',
            'MLP',
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
                'MLP',
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