import sys
sys.path.append('../')
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import pyro.poutine as poutine
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    input_size: int = 4
    hidden_size: int = 8
    z_dim: int = 32
    emission_dim: int = 32
    transition_dim: int = 32
    rnn_dim: int = 32
    output_size: int = 1
    variance: float = 0.1

class Emitter(nn.Module):
    """
    Parameterizes the Gaussian observation likelihood p(y_t | z_t, x_t)
    """
    def __init__(self,
                 args: ModelArgs
                 ):

        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(args.z_dim, args.emission_dim)
        self.lin_x_to_hidden = nn.Linear(args.input_size, args.emission_dim)
        self.lin_hidden_to_mean = nn.Linear(args.emission_dim, 1)
        # initialize the fixed variance hyperparameter
        self.variance = nn.Parameter(torch.tensor(args.variance))
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
    def __init__(self, 
                 args: ModelArgs
                 ):
        super().__init__()
        # initialize the two linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(args.z_dim, 
                                         args.transition_dim)
        self.lin_x_to_hidden = nn.Linear(args.input_size, 
                                         args.transition_dim)
        # initialize the two gated transformations used in the neural network
        self.lin_hidden_to_hidden1 = nn.Linear(args.transition_dim, 
                                               args.transition_dim)
        self.lin_hidden_to_hidden2 = nn.Linear(args.transition_dim, 
                                               args.transition_dim)
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
    def __init__(self, 
                 args: ModelArgs
                 ):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(args.z_dim, 
                                         args.emission_dim)
        self.lin_rnn_to_hidden = nn.Linear(args.rnn_dim, 
                                           args.emission_dim)
        self.hidden_to_loc = nn.Linear(args.emission_dim, 
                                       args.z_dim)
        self.hidden_to_scale = nn.Linear(args.emission_dim, 
                                         args.z_dim)
        self.relu = nn.ReLU()

    def forward(self, z_prev, rnn_input):
        hidden = self.relu(self.lin_z_to_hidden(z_prev) + self.lin_rnn_to_hidden(rnn_input))
        loc = self.hidden_to_loc(hidden)
        scale = F.softplus(self.hidden_to_scale(hidden))
        return loc, scale
    
class _DeepMarkovModel(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """
    def __init__(self, 
                args: ModelArgs
                ):
        
        super().__init__()

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(args)
        self.transition = GatedTransition(args)
        self.combiner = Combiner(args)

        self.rnn = nn.GRU(input_size=args.input_size, 
                          hidden_size=args.rnn_dim,
                          batch_first=True,
                          bidirectional=False, 
                          num_layers=2
                          )

        if use_cuda:
            if torch.cuda.device_count() > 1:
                self.emitter = nn.DataParallel(self.emitter)
                self.transition = nn.DataParallel(self.transition)
                self.combiner = nn.DataParallel(self.combiner)
                self.rnn = nn.DataParallel(self.rnn)

            self.emitter = self.emitter.to(device)
            self.transition = self.transition.to(device)
            self.combiner = self.combiner.to(device)
            self.rnn = self.rnn.to(self.device)

        # define a (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(args.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(1, args.z_dim))
        # define a (trainable) parameter for the initial hidden state of the RNN
        self.h_0 = nn.Parameter(torch.zeros(1, 1, args.rnn_dim))

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
        
        z_q_0_expanded = self.z_q_0.expand(x_data.size(0), ModelArgs.z_dim)
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

    @property
    def model_type(self):
        return "probabilistic"