import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from stockpy.base import Regressor
from stockpy.base import Classifier 

from ._emitter import EmitterClassifier, EmitterRegressor
from ._transition import Transition

__all__ = ['NNHMMRegressor', 'NNHMMClassifier']

class NNHMM(PyroModule):
    """
    A Gaussian Hidden Markov Model (NNHMM) class that encapsulates neural network
    components for modeling sequences with latent variables. The model can be used
    for both classification and regression tasks.

    Parameters
    ----------
    z_dim : int
        Dimension of the latent variable `z_t`.
    emission_dim : int
        Size of the hidden layer for the emission model.
    transition_dim : int
        Size of the hidden layer for the transition model.
    dropout : float
        Dropout rate for regularization.
    variance : float
        Variance for the latent state initialization.
    activation : str
        Name of the activation function to be used.
    bias : bool
        Flag to include bias in linear layers or not.
    seq_len : int
        Length of the sequences to model.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    z_dim : int
        Dimensionality of the latent state.
    emission_dim : int
        Dimensionality of the hidden layers in the emission model.
    transition_dim : int
        Dimensionality of the hidden layers in the transition model.
    dropout : float
        Dropout rate used in the neural network components.
    variance : float
        The variance used in the latent state initialization.
    activation : str
        The type of activation function to use.
    bias : bool
        Whether to include bias parameters in the neural network layers.
    seq_len : int
        The length of the input sequences.
    z_0 : nn.Parameter
        Initial latent state parameter for `z_1`.
    z_q_0 : nn.Parameter
        Initial latent state parameter for `q(z_1)`.
    emitter_rgr : EmitterRegressor
        Emission model for regression tasks.
    emitter_cls : EmitterClassifier
        Emission model for classification tasks.
    transition : Transition
        Transition model defining the evolution of latent states.
    """

    def __init__(self,
                 z_dim=32,
                 emission_dim=32,
                 transition_dim=32,
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
        self.dropout = dropout
        self.variance = variance
        self.activation = activation
        self.bias = bias
        self.seq_len = seq_len

    def initialize_module(self):
        """
        Initializes the various neural network components of the NNHMM based on the configured parameters.

        This method is responsible for initializing the NNHMM-specific parameters, including the initial state probabilities, 
        the transition matrix, the emission means and variances, and the initial latent state parameters `z_0` and `z_q_0`.

        If the NNHMM is being used for classification, the output size is set to the number of classes. If it's being used for 
        regression, the output size is set to the number of outputs.

        Notes
        -----
        This method modifies the NNHMM in-place, initializing its various components.

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
                
        # define a (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(1, self.z_dim))

    @property
    def model_type(self):
        return "rnn"
    
class NNHMMRegressor(Regressor, NNHMM):
    """
    A regressor based on Gaussian Hidden Markov Models (GHMMs) which combines the properties
    of a NNHMM with the capabilities of a regressor. It can be used to perform sequence-based
    regression tasks, predicting continuous outputs from input sequences.

    Inherits from:
        Regressor: A base regression class that may define general regression functionalities.
        NNHMM: A Gaussian Hidden Markov Model class with sequence modeling capabilities.

    Parameters
    ----------
    z_dim : int
        Dimension of the latent variable `z_t`.
    emission_dim : int
        Size of the hidden layer for the emission model.
    transition_dim : int
        Size of the hidden layer for the transition model.
    dropout : float
        Dropout rate for regularization.
    variance : float
        Variance for the latent state initialization.
    activation : str
        Name of the activation function to be used.
    bias : bool
        Flag to include bias in linear layers or not.
    seq_len : int
        Length of the sequences to model.
    **kwargs
        Additional keyword arguments specific to the Regressor base class.

    Attributes
    ----------
    Inherits attributes from NNHMM:
        z_dim : int
            Dimensionality of the latent state.
        emission_dim : int
            Dimensionality of the hidden layers in the emission model.
        transition_dim : int
            Dimensionality of the hidden layers in the transition model.
        dropout : float
            Dropout rate used in the neural network components.
        variance : float
            The variance used in the latent state initialization.
        activation : str
            The type of activation function to use.
        bias : bool
            Whether to include bias parameters in the neural network layers.
        seq_len : int
            The length of the input sequences.
        z_0 : nn.Parameter
            Initial latent state parameter for `z_1`.
        z_q_0 : nn.Parameter
            Initial latent state parameter for `q(z_1)`.
        emitter_rgr : EmitterRegressor
            Emission model for regression tasks.
        transition : Transition
            Transition model defining the evolution of latent states.

    Note
    -----
        The initialization parameters will first initialize the `Regressor` base class
        with provided keyword arguments, then initialize the `NNHMM` class with specified
        parameters for the NNHMM components.
    """

    def __init__(self,
                 z_dim=32,
                 emission_dim=32,
                 transition_dim=32,
                 dropout=0.2,
                 variance=0.1,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 **kwargs):
        """
        Constructor for the NNHMMRegressor class, initializing the components of both
        the Regressor and the NNHMM classes with the given parameters.

        The **kwargs are passed directly to the base Regressor class to allow for 
        flexibility in configuring any additional regressor-specific settings.
        """

        Regressor.__init__(self, **kwargs)
        NNHMM.__init__(self,
                 z_dim=z_dim,
                 emission_dim=emission_dim,
                 transition_dim=transition_dim,
                 dropout=dropout,
                 variance=variance,
                 activation=activation,
                 bias=bias,
                 seq_len=seq_len,
                 **kwargs)

    def model(self, x, y, annealing_factor=1.0):
        """
        The generative model for the variational autoencoder, which specifies the joint
        probability distribution over the latent variables `z` and observed variables `y`
        given the inputs `x`.

        Parameters
        ----------
        x : torch.Tensor
            The input features with shape (batch_size, seq_len, num_features).
        y : torch.Tensor
            The target values with shape (batch_size, seq_len, output_size).
        annealing_factor : float, optional
            A scaling factor for the KL divergence term to control its influence on the loss.
            Defaults to 1.0.

        Note
        ----
        This method makes use of the Pyro's plate notation to efficiently handle mini-batches,
        and markov to indicate conditional independencies that arise in markov models, allowing
        for more efficient inference.
        """

        # Number of time steps to process
        T_max = x.size(1)

        # Register PyTorch (sub)modules with Pyro
        pyro.module("nhmm", self)

        # Expand the initial latent state to match the batch size
        z_prev = self.z_0.expand(x.size(0), self.z_0.size(0))

        # Process the sequence one time step at a time
        with pyro.plate("z_minibatch", len(x)):
            for t in pyro.markov(range(1, T_max + 1)):

                # Compute the parameters of the latent state distribution at time t
                z_loc, z_scale = self.transition(z_prev, x[:, t - 1, :])

                # Ensure the scale is positive
                z_scale = z_scale.clamp(min=1e-6)

                # Sample the latent variable z_t using the reparameterization trick
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(f"z_{t}",
                                      dist.Normal(z_loc, z_scale).to_event(1))

                # Compute the parameters of the observation distribution at time t
                mu, sigma = self.emitter_rgr(z_t, x[:, t - 1, :])

                # Sample the observation y at time t
                pyro.sample(f"obs_y_{t}",
                            dist.Normal(mu, sigma).to_event(1),
                            obs=y)

                # Update the previous latent state
                z_prev = z_t
            
    def guide(self, x, y=None, annealing_factor=1.0):
        """
        The variational guide (approximate posterior) for the variational autoencoder,
        which defines the family of distributions over the latent variables `z` that will
        be optimized to approximate the true posterior.

        Parameters
        ----------
        x : torch.Tensor
            The input features with shape (batch_size, seq_len, num_features).
        y : torch.Tensor, optional
            The target values. This is not used in the guide and is included to match the
            signature of the model.
        annealing_factor : float, optional
            A scaling factor for the KL divergence term. Defaults to 1.0.

        Note
        ----
        This guide corresponds to the mean-field approximation where the latent variables
        at each time step are independent given the observed data.
        """

        # Number of time steps to process
        T_max = x.size(1)

        # Register PyTorch (sub)modules with Pyro
        pyro.module("nhmm", self)

        # Expand the initial latent state to match the batch size
        z_prev = self.z_q_0.expand(x.size(0), self.z_dim)

        with pyro.plate("z_minibatch", len(x)):
            for t in pyro.markov(range(1, T_max + 1)):
                # Compute the parameters of the guide distribution for z_t
                z_loc, z_scale = self.transition(z_prev, x[:, t - 1, :])

                # Ensure the scale is positive
                z_scale = z_scale.clamp(min=1e-6)

                # Define the distribution q(z_t | z_{t-1}, x_{t:T})
                z_dist = dist.Normal(z_loc, z_scale)

                # Sample z_t from the guide distribution
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(f"z_{t}", z_dist.to_event(1))

                # Update the previous latent state
                z_prev = z_t 

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the neural network.

        Returns
        -------
        torch.Tensor
            Output tensor from the neural network.
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
                
class NNHMMClassifier(Classifier, NNHMM):
    """
    A classifier based on Gaussian Hidden Markov Models (GHMMs) which combines the properties
    of a NNHMM with the capabilities of a classifier. It can be used to perform sequence-based
    classification tasks, predicting discrete outputs from input sequences.

    Inherits from:
        Classifier: A base classification class that may define general classification functionalities.
        NNHMM: A Gaussian Hidden Markov Model class with sequence modeling capabilities.

    Parameters
    ----------
    z_dim : int
        Dimension of the latent variable `z_t`.
    emission_dim : int
        Size of the hidden layer for the emission model.
    transition_dim : int
        Size of the hidden layer for the transition model.
    dropout : float
        Dropout rate for regularization.
    variance : float
        Variance for the latent state initialization.
    activation : str
        Name of the activation function to be used.
    bias : bool
        Flag to include bias in linear layers or not.
    seq_len : int
        Length of the sequences to model.
    **kwargs
        Additional keyword arguments specific to the Classifier base class.

    Attributes
    ----------
    Inherits attributes from NNHMM:
        z_dim : int
            Dimensionality of the latent state.
        emission_dim : int
            Dimensionality of the hidden layers in the emission model.
        transition_dim : int
            Dimensionality of the hidden layers in the transition model.
        dropout : float
            Dropout rate used in the neural network components.
        variance : float
            The variance used in the latent state initialization.
        activation : str
            The type of activation function to use.
        bias : bool
            Whether to include bias parameters in the neural network layers.
        seq_len : int
            The length of the input sequences.
        z_0 : nn.Parameter
            Initial latent state parameter for `z_1`.
        z_q_0 : nn.Parameter
            Initial latent state parameter for `q(z_1)`.
        emitter_cls : EmitterClassifier
            Emission model for classification tasks.
        transition : Transition
            Transition model defining the evolution of latent states.

    Note
    -----
    The initialization parameters will first initialize the `Classifier` base class
    with provided keyword arguments, then initialize the `NNHMM` class with specified
    parameters for the NNHMM components.
    """

    def __init__(self,
                 z_dim=32,
                 emission_dim=32,
                 transition_dim=32,
                 dropout=0.2,
                 variance=0.1,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 **kwargs):
        """
        Constructor for the GHMMClassifier class, initializing the components of both
        the Classifier and the NNHMM classes with the given parameters.

        The **kwargs are passed directly to the base Classifier class to allow for 
        flexibility in configuring any additional classifier-specific settings.
        """
        Classifier.__init__(self, **kwargs)
        NNHMM.__init__(self,
                 z_dim=z_dim,
                 emission_dim=emission_dim,
                 transition_dim=transition_dim,
                 dropout=dropout,
                 variance=variance,
                 activation=activation,
                 bias=bias,
                 seq_len=seq_len,
                 **kwargs)
        
    def model(self, x, y, annealing_factor=1.0):
        """
        Defines the generative model p(y,z|x) which includes the observation 
        model p(y|z) and the transition model p(z_t | z_{t-1}). It also handles the 
        computation of the parameters of these models.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the model.
        y : Optional[torch.Tensor]
            Optional observed output tensor for the model.
        annealing_factor : float, optional
            Annealing factor used in poutine.scale to handle KL annealing, by default 1.0.

        Returns
        -------
        torch.Tensor
            The sampled latent variable `z` from the last time step of the model.
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

                z_scale = z_scale.clamp(min=1e-6)
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
            
    def guide(self, x, y=None, annealing_factor=1.0):
        """
        Defines the guide (also called the inference model or variational distribution) q(z|x,y)
        which is an approximation to the posterior p(z|x,y). It also handles the computation of the 
        parameters of this guide.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the guide.
        y : Optional[torch.Tensor], optional
            Optional observed output tensor for the guide.
        annealing_factor : float, optional
            Annealing factor used in poutine.scale to handle KL annealing, by default 1.0.

        Returns
        -------
        torch.Tensor
            The sampled latent variable `z` from the last time step of the guide.
        """
        
        # this is the number of time steps we need to process in the mini-batch
        T_max = x.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # initialize the values of the latent variables using heuristics        
        z_prev = self.z_q_0.expand(x.size(0), self.z_dim)
        
        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(x)):
            # sample the latents z one time step at a time
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})

                z_loc, z_scale = self.transition(z_prev, x[:, t - 1, :])

                z_scale = z_scale.clamp(min=1e-6)

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

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the neural network.

        Returns
        -------
        torch.Tensor
            Output tensor from the neural network.
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