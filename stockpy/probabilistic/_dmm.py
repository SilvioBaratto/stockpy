import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from stockpy.base import Regressor
from stockpy.base import Classifier 
from stockpy.utils import to_device
from stockpy.utils import get_activation_function

from ._combiner import Combiner
from ._emitter import EmitterRegressor, EmitterClassifier
from ._transition import Transition

class DMM(PyroModule):

    def __init__(self,
                 z_dim=32,
                 emission_dim=32,
                 transition_dim=32,
                 rnn_dim=32,
                 num_layers=1,
                 dropout=0.2,
                 variance=0.1,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 **kwargs):
        
        super().__init__()

        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.variance = variance
        self.activation = activation
        self.bias = bias
        self.seq_len = seq_len

    def reset_weights(self):
        """
        Reinitializes the weights of the neural network.
        """
        pass

    def initialize_module(self):
        """
        Initializes the layers of the neural network based on configuration.
        """

        if isinstance(self, Classifier):
            self.output_size = self.n_classes_
        elif isinstance(self, Regressor):
            self.output_size = self.n_outputs_

        self.emitter_rgr = EmitterRegressor(self.n_features_in_,
                                        self.z_dim, 
                                        self.emission_dim,
                                        self.output_size)

        self.emitter_cls = EmitterClassifier(self.n_features_in_,
                                        self.z_dim, 
                                        self.emission_dim,
                                        self.output_size)

        self.transition = Transition(self.z_dim, 
                                     self.n_features_in_, 
                                     self.transition_dim)
        
        self.combiner = Combiner(self.z_dim, self.rnn_dim)
        
        self.rnn = nn.GRU(input_size=self.n_features_in_,
                            hidden_size=self.rnn_dim,
                            batch_first=True,
                            bidirectional=False,
                            num_layers=self.num_layers,
                            bias=self.bias,
                            )
        
        to_device(self.emitter_rgr, self.device)
        to_device(self.emitter_cls, self.device)
        to_device(self.transition, self.device)
        to_device(self.combiner, self.device)
        to_device(self.rnn, self.device)
        
        # define a (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(1, self.z_dim))
        # define a (trainable) parameter for the initial hidden state of the RNN
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn_dim))

    @property
    def model_type(self):
        return "rnn"
    
class DMMRegressor(Regressor, DMM):

    def __init__(self,
                 z_dim=32,
                 emission_dim=32,
                 transition_dim=32,
                 rnn_dim=32,
                 num_layers=1,
                 dropout=0.2,
                 variance=0.1,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 **kwargs):
        """
        Initializes the MLPClassifier object with given or default parameters.
        """
        Regressor.__init__(self, **kwargs)
        DMM.__init__(self,
                     z_dim=z_dim,
                     emission_dim=emission_dim,
                     transition_dim=transition_dim,
                     rnn_dim=rnn_dim,
                     num_layers=num_layers,
                     dropout=dropout,
                     variance=variance,
                     activation=activation,
                     bias=bias,
                     seq_len=seq_len,
                     **kwargs)
        
        self.criterion = nn.MSELoss()

    def model(self, x, y, annealing_factor = 1.0):
        """
        Defines the generative model p(y,z|x) which includes the observation 
        model p(y|z) and transition model p(z_t | z_{t-1}). It also handles the 
        computation of the parameters of these models.

        Args:
            x_data (torch.Tensor): Input tensor for the model.
            y_data (Optional[torch.Tensor]): Optional observed output tensor for the model.
            annealing_factor (float, optional): Annealing factor used in poutine.scale to handle KL annealing.

        Returns:
            torch.Tensor: The sampled latent variable `z` from the last time step of the model.
        """

        # this is the number of time steps we need to process in the data
        T_max = x.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(x.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(x)):
            # sample the latents z and observed y's one time step at a time
            for t in pyro.markov(range(1, T_max + 1)):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing.

                # first compute the parameters of the diagonal gaussian
                # distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.transition(z_prev, x[:, t - 1, :])

                # then sample z_t according to dist.Normal(z_loc, z_scale).
                # note that we use the reshape method so that the univariate
                # Normal distribution is treated as a multivariate Normal
                # distribution with a diagonal covariance.

                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t,
                                    dist.Normal(z_loc, z_scale)
                                                .to_event(1))

                # compute the mean of the Gaussian distribution p(y_t | z_t)
                mu, sigma = self.emitter_rgr(z_t, x[:, t - 1, :])

                # the next statement instructs pyro to observe y_t according to the
                # Gaussian distribution p(y_t | z_t)
                pyro.sample("obs_y_%d" % t,
                            dist.Normal(mu, sigma).to_event(1),
                            obs=y)
                
                # print(f"model z_prev shape: {z_prev.shape}")
                # print(f"model x_data[:, t - 1, :] shape: {x_data[:, t - 1, :].shape}")   
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t
            
    def guide(self, x, y = None, annealing_factor = 1.0):
        """
        Defines the guide (also called the inference model or variational distribution) q(z|x,y)
        which is an approximation to the posterior p(z|x,y). It also handles the computation of the 
        parameters of this guide.

        Args:
            x_data (torch.Tensor): Input tensor for the guide.
            y_data (Optional[torch.Tensor]): Optional observed output tensor for the guide.
            annealing_factor (float, optional): Annealing factor used in poutine.scale to handle KL annealing.

        Returns:
            torch.Tensor: The sampled latent variable `z` from the last time step of the guide.
        """
        
        # this is the number of time steps we need to process in the mini-batch
        T_max = x.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)
        
        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(self.num_layers, x.size(0), self.rnn.hidden_size).contiguous()
        
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(x, h_0_contig)

        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = rnn_output.flip(1)
        
        z_prev = self.z_q_0.expand(x.size(0), self.z_dim)
        
        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(x)):
            # sample the latents z one time step at a time
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})

                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                z_dist = dist.Normal(z_loc, z_scale)

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(None, annealing_factor):
                    z_t = pyro.sample("z_%d" % t, z_dist.to_event(1))
           
                # the latent sampled at this time step will be conditioned
                # upon in the next time step so keep track of it
                z_prev = z_t  

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor for the neural network.

        Returns:
            torch.Tensor: Output tensor from the neural network.
        """
        preds = []
                
        # Run the guide and capture the trace
        guide_trace = pyro.poutine.trace(self.guide).get_trace(x)

        # This is the number of time steps we need to process in the mini-batch
        T_max = x.size(1)

        # Extract the latent variables from the trace and pass them to the emitter
        for t in pyro.markov(range(1, T_max + 1)):

            z_t = guide_trace.nodes[f"z_{t}"]["value"]

            mean_t, _ = self.emitter_rgr(z_t, x[:, t - 1, :])
            
            preds.append(mean_t)
                
        # Stack and average the predictions
        preds = torch.stack(preds)

        return preds[-1, :, :]
                
class DMMClassifier(Classifier, DMM):

    def __init__(self,
                 z_dim=32,
                 emission_dim=32,
                 transition_dim=32,
                 rnn_dim=32,
                 num_layers=1,
                 dropout=0.2,
                 variance=0.1,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 **kwargs):
        """
        Initializes the MLPClassifier object with given or default parameters.
        """
        Classifier.__init__(self, **kwargs)
        DMM.__init__(self,
                     z_dim=z_dim,
                     emission_dim=emission_dim,
                     transition_dim=transition_dim,
                     rnn_dim=rnn_dim,
                     num_layers=num_layers,
                     dropout=dropout,
                     variance=variance,
                     activation=activation,
                     bias=bias,
                     seq_len=seq_len,
                     **kwargs)
        
        self.criterion = nn.NLLLoss()

    def model(self, x, y, annealing_factor = 1.0):
        """
        Defines the generative model p(y,z|x) which includes the observation 
        model p(y|z) and transition model p(z_t | z_{t-1}). It also handles the 
        computation of the parameters of these models.

        Args:
            x_data (torch.Tensor): Input tensor for the model.
            y_data (Optional[torch.Tensor]): Optional observed output tensor for the model.
            annealing_factor (float, optional): Annealing factor used in poutine.scale to handle KL annealing.

        Returns:
            torch.Tensor: The sampled latent variable `z` from the last time step of the model.
        """

        # this is the number of time steps we need to process in the data
        T_max = x.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(x.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(x)):
            # sample the latents z and observed y's one time step at a time
            for t in pyro.markov(range(1, T_max + 1)):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing.

                # first compute the parameters of the diagonal gaussian
                # distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.transition(z_prev, x[:, t - 1, :])

                # then sample z_t according to dist.Normal(z_loc, z_scale).
                # note that we use the reshape method so that the univariate
                # Normal distribution is treated as a multivariate Normal
                # distribution with a diagonal covariance.

                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t,
                                    dist.Normal(z_loc, z_scale)
                                                .to_event(1))

                # compute the mean of the Gaussian distribution p(y_t | z_t)
                probs_t = self.emitter_cls(z_t, x[:, t - 1, :])

                # the next statement instructs pyro to observe y_t according to the
                # Gaussian distribution p(y_t | z_t)
                pyro.sample("obs_y_%d" % t,
                            dist.Categorical(probs=probs_t).to_event(1),obs=y)
                
                # print(f"model z_prev shape: {z_prev.shape}")
                # print(f"model x_data[:, t - 1, :] shape: {x_data[:, t - 1, :].shape}")   
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t
            
    def guide(self, x, y = None, annealing_factor = 1.0):
        """
        Defines the guide (also called the inference model or variational distribution) q(z|x,y)
        which is an approximation to the posterior p(z|x,y). It also handles the computation of the 
        parameters of this guide.

        Args:
            x_data (torch.Tensor): Input tensor for the guide.
            y_data (Optional[torch.Tensor]): Optional observed output tensor for the guide.
            annealing_factor (float, optional): Annealing factor used in poutine.scale to handle KL annealing.

        Returns:
            torch.Tensor: The sampled latent variable `z` from the last time step of the guide.
        """
        
        # this is the number of time steps we need to process in the mini-batch
        T_max = x.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)
        
        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(self.num_layers, x.size(0), self.rnn.hidden_size).contiguous()
        
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(x, h_0_contig)

        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = rnn_output.flip(1)
        
        z_prev = self.z_q_0.expand(x.size(0), self.z_dim)
        
        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(x)):
            # sample the latents z one time step at a time
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})

                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                z_dist = dist.Normal(z_loc, z_scale)

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(None, annealing_factor):
                    z_t = pyro.sample("z_%d" % t, z_dist.to_event(1))
           
                # the latent sampled at this time step will be conditioned
                # upon in the next time step so keep track of it
                z_prev = z_t 

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor for the neural network.

        Returns:
            torch.Tensor: Output tensor from the neural network.
        """
        preds = []
                
        # Run the guide and capture the trace
        guide_trace = pyro.poutine.trace(self.guide).get_trace(x)

        # This is the number of time steps we need to process in the mini-batch
        T_max = x.size(1)

        # Extract the latent variables from the trace and pass them to the emitter
        for t in pyro.markov(range(1, T_max + 1)):

            z_t = guide_trace.nodes[f"z_{t}"]["value"]
                
            class_logits_t = self.emitter_cls(z_t, x[:, t - 1, :])
            
            preds.append(class_logits_t)
                
        # Stack and average the predictions
        preds = torch.stack(preds)

        return preds.mean(0)