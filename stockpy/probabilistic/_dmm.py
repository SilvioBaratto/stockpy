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

    """
    Deep Markov Model (DMM) implemented as a PyroModule, representing a class of deep generative models
    where the latent state space follows a Markov process.

    Attributes:
        z_dim (int): Dimensionality of the latent state space.
        emission_dim (int): Dimensionality of the emission parameters.
        transition_dim (int): Dimensionality of the transition parameters.
        rnn_dim (int): Dimensionality of the RNN's hidden states.
        num_layers (int): Number of layers in the RNN.
        dropout (float): Dropout rate for regularization.
        variance (float): Variance parameter for certain probability distributions.
        activation (str): Activation function used in neural network layers.
        bias (bool): If `True`, layers will use bias terms.
        seq_len (int): Length of the input sequences.
        emitter_rgr (PyroModule): The regression module for emission.
        emitter_cls (PyroModule): The classification module for emission.
        transition (PyroModule): The transition module.
        combiner (PyroModule): Combines RNN outputs with latent states.
        rnn (torch.nn.Module): Recurrent neural network module.
        z_0 (torch.nn.Parameter): Initial latent state parameter.
        z_q_0 (torch.nn.Parameter): Initial parameter for the variational distribution of the latent state.
        h_0 (torch.nn.Parameter): Initial hidden state parameter for the RNN.

    Args:
        z_dim (int, optional): Dimensionality of the latent state space (default: 32).
        emission_dim (int, optional): Dimensionality of the emission parameters (default: 32).
        transition_dim (int, optional): Dimensionality of the transition parameters (default: 32).
        rnn_dim (int, optional): Dimensionality of the RNN's hidden states (default: 32).
        num_layers (int, optional): Number of layers in the RNN (default: 1).
        dropout (float, optional): Dropout rate for regularization (default: 0.2).
        variance (float, optional): Variance parameter for certain probability distributions (default: 0.1).
        activation (str, optional): Activation function used in neural network layers (default: 'relu').
        bias (bool, optional): If `True`, layers will use bias terms (default: True).
        seq_len (int, optional): Length of the input sequences (default: 20).
        **kwargs: Arbitrary keyword arguments.

    Note:
        Inherits from PyroModule to seamlessly integrate deep learning with probabilistic modeling.
        The `DMM` can operate as both a regressor and a classifier depending on the type of the
        emitter module initialized.
    """

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

    def initialize_module(self):
        """
        Initializes the neural network modules and parameters of the DMM.
        This method should be called after creating an instance of DMM and before
        using it for inference or training.

        Raises:
            TypeError: If the object does not have the expected attributes due to not
                       being an instance of `Classifier` or `Regressor`.
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

    """
    Deep Markov Model Regressor (DMMRegressor) is a specialized version of the DMM that is designed
    to perform regression tasks using deep generative modeling.

    Inherits from:
        Regressor: A base class for regression models.
        DMM: The base Deep Markov Model class for deep generative models.

    Attributes:
        Inherits all attributes from the DMM class and potentially modifies output dimensions
        based on the regression requirements.

    Args:
        z_dim (int, optional): Dimensionality of the latent state space (default: 32).
        emission_dim (int, optional): Dimensionality of the emission parameters (default: 32).
        transition_dim (int, optional): Dimensionality of the transition parameters (default: 32).
        rnn_dim (int, optional): Dimensionality of the RNN's hidden states (default: 32).
        num_layers (int, optional): Number of layers in the RNN (default: 1).
        dropout (float, optional): Dropout rate for regularization (default: 0.2).
        variance (float, optional): Variance parameter for certain probability distributions (default: 0.1).
        activation (str, optional): Activation function used in neural network layers (default: 'relu').
        bias (bool, optional): If `True`, adds bias to RNN layers (default: True).
        seq_len (int, optional): Length of the input sequences (default: 20).
        **kwargs: Arbitrary keyword arguments that are passed to the Regressor base class.

    Note:
        The `DMMRegressor` initializes the DMM as a regression model, adjusting the output
        dimensions to match the requirements of the regression task.
    """

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
        Constructor method for the DMMRegressor class.
        
        This method initializes a new instance of DMMRegressor with the specified
        parameters or their default values. It first initializes the Regressor base class
        and then the DMM class with the provided arguments.

        Raises:
            TypeError: If an argument passed is not of the expected type.
            ValueError: If an invalid value is passed to an argument.
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
        Define the generative model for a deep Markov model (DMM) regressor.

        The model is a Pyro probabilistic model, which includes the definition of the prior over
        the latent variables z and the likelihood of the observations y given z. It is used for
        training the model with variational inference.

        Args:
            x (torch.Tensor): The observed input features with shape (batch_size, T_max, input_dim),
                            where T_max is the maximum sequence length in the batch.
            y (torch.Tensor): The target variable with shape (batch_size, T_max, output_dim),
                            where T_max is the maximum sequence length in the batch.
            annealing_factor (float, optional): Factor to anneal the KL-divergence term in the loss
                                                during training (default is 1.0).

        Notes:
            - The `model` function is part of the Pyro model-guide pair required for SVI.
            - It includes annealing of the KL-divergence to stabilize training in the early epochs.
            - The `pyro.plate` construct is used to denote independent batches during the sampling process.
            - Time dependencies are handled through Pyro's `markov` context manager.
        """

        # Determine the sequence length from the input features `x`.
        T_max = x.size(1)

        # Register this module with Pyro so that all its parameters can be optimized.
        pyro.module("dmm", self)

        # Initialize the previous latent state `z_prev` with `z_0`, which is a trainable parameter.
        # This is for the first time step where there's no previous latent state available.
        z_prev = self.z_0.expand(x.size(0), self.z_0.size(0))

        # Start a Pyro plate for batching. This handles the mini-batch processing within Pyro,
        # allowing for vectorized operations across the batch dimension.
        with pyro.plate("z_minibatch", len(x)):
            # Loop over each time step in the sequence. Pyro's markov context manager tells Pyro
            # that the current time step only depends on the previous one, which enables 
            # optimizations for markov models.
            for t in pyro.markov(range(1, T_max + 1)):

                # Get the parameters for the distribution of the current latent state `z_t`
                # by passing the previous latent state and the current observed data through
                # the transition module.
                z_loc, z_scale = self.transition(z_prev, x[:, t - 1, :])

                # Scale the contribution of the KL-divergence to the loss function by the annealing factor.
                # This can help in stabilizing training by initially weighting the KL-divergence term less.
                with pyro.poutine.scale(scale=annealing_factor):
                    # Sample the latent variable `z_t` for time step `t` from the normal distribution
                    # parameterized by `z_loc` and `z_scale`. `.to_event(1)` indicates that this
                    # distribution is over a vector-valued random variable.
                    z_t = pyro.sample("z_%d" % t, dist.Normal(z_loc, z_scale).to_event(1))

                # Pass the sampled latent state `z_t` and the observed data to the emission module
                # to get the parameters `mu` and `sigma` of the observed distribution.
                mu, sigma = self.emitter_rgr(z_t, x[:, t - 1, :])

                # Sample the observation `y` at time `t` from the normal distribution parameterized
                # by the output of the emission module. The `obs=y[:, t - 1, :]` argument indicates
                # that this sample corresponds to the actual observed data.
                pyro.sample("obs_y_%d" % t, dist.Normal(mu, sigma).to_event(1), obs=y[:, t - 1, :])

                # Update `z_prev` to the current `z_t` to be used in the next time step.
                z_prev = z_t
            
    def guide(self, x, y = None, annealing_factor = 1.0):

        """
        Define the variational guide for the deep Markov model (DMM) regressor.

        The guide serves as an approximate posterior that is optimized to resemble the true posterior
        of the latent variables given the observations. It is a parametrized distribution from which
        we can sample z and is defined using the same plates as the model for the latent variables.

        Args:
            x (torch.Tensor): The observed input features with shape (batch_size, T_max, input_dim),
                            where T_max is the maximum sequence length in the batch.
            y (torch.Tensor, optional): The target variable is not used in the guide and included
                                        for API consistency (default is None).
            annealing_factor (float, optional): Factor to anneal the KL-divergence term in the loss
                                                during training (default is 1.0).

        Notes:
            - The `guide` function is part of the Pyro model-guide pair required for SVI.
            - It specifies the family of distributions used for approximation of the posterior.
            - Time dependencies are handled through Pyro's `markov` context manager.
            - The hidden state of the RNN is used to compute the parameters of the variational distribution.
        """
        
        # Determine the sequence length from the input features `x`.
        T_max = x.size(1)
        # Register this module with Pyro, which is necessary for optimization.
        pyro.module("dmm", self)
        
        # Prepare the initial hidden state for the RNN, ensuring it's compatible with the input's batch size.
        h_0_contig = self.h_0.expand(self.num_layers, x.size(0), self.rnn.hidden_size).contiguous()
        
        # Process the sequence `x` through the RNN to obtain the output for each time step.
        rnn_output, _ = self.rnn(x, h_0_contig)

        # Invert the sequence of the RNN output for backwards time processing if needed.
        rnn_output = rnn_output.flip(1)
        
        # Initialize the previous latent state `z_prev` with `z_q_0`, which is a trainable parameter
        # that represents the initial state of the latent variable.
        z_prev = self.z_q_0.expand(x.size(0), self.z_dim)
        
        # Start a Pyro plate for batching, similar to the model function.
        with pyro.plate("z_minibatch", len(x)):
            # Loop over each time step in the sequence in reverse order.
            for t in pyro.markov(range(1, T_max + 1)):

                # Use the `combiner` to calculate the parameters of the variational distribution for `z_t`
                # based on the previous latent state `z_prev` and the output of the RNN at the corresponding time step.
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                # Define the normal distribution for the current latent variable `z_t` with the computed mean and scale.
                z_dist = dist.Normal(z_loc, z_scale)

                # Sample `z_t` from the variational distribution without scaling the KL-divergence term.
                # This is in contrast with the model where the annealing factor scales the KL-divergence term.
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, z_dist.to_event(1))

                # Update `z_prev` with the sampled `z_t` to be used for the next time step.
                z_prev = z_t

    def forward(self, x):
        """
        Forward pass for the Deep Markov Model (DMM) regressor to generate predictions.

        Given an input sequence `x`, this method produces the sequence of predictions by
        performing the following steps for each time step:
        1. Retrieves the latent variables `z_t` from the guide trace.
        2. Generates the mean predictions from the emitter regressor using `z_t`.

        The predictions for all time steps are then combined, and the output for the last
        time step is returned as the final prediction for the sequence.

        Args:
            x (torch.Tensor): The input tensor for the neural network with shape
                            (batch_size, T_max, input_dim), where `T_max` is the maximum
                            sequence length in the batch, and `input_dim` is the dimension
                            of the input features.

        Returns:
            torch.Tensor: The output tensor from the neural network containing the predicted
                        values for the last time step in the input sequence. The shape of
                        the tensor is (batch_size, output_dim), where `output_dim` is the
                        dimension of the output features.

        Notes:
            - This method assumes that `guide` has been defined and can provide the latent
            variables `z_t` required for generating the predictions.
            - The predictions for only the last time step are returned, which aligns with
            many sequence-to-one prediction tasks. To get predictions for all time steps,
            modify the method to return the entire `preds` tensor.
        """
        # Initialize a list to hold the predictions at each time step.
        preds = []
                
        # Record the operations of the guide function on the input `x` to later access the latent variables.
        guide_trace = pyro.poutine.trace(self.guide).get_trace(x)

        # Determine the sequence length from the input features `x`.
        T_max = x.size(1)

        # Loop over each time step in the mini-batch.
        for t in pyro.markov(range(1, T_max + 1)):

            # Retrieve the value of the latent variable `z_t` from the guide trace for the current time step.
            z_t = guide_trace.nodes[f"z_{t}"]["value"]

            # Use the emitter (which is typically a neural network module) to get the mean prediction at time `t`.
            # The underscore '_' is used to discard the second return value (typically the standard deviation).
            mean_t, _ = self.emitter_rgr(z_t, x[:, t - 1, :])
            
            # Append the mean prediction for the current time step to the list of predictions.
            preds.append(mean_t)
                
        # Combine the predictions from all time steps into a single tensor.
        preds = torch.stack(preds)

        # Return the predictions for the last time step for each element in the batch.
        return preds[-1, :, :]
                
class DMMClassifier(Classifier, DMM):

    """
    DMMClassifier is a deep Markov model for classification tasks.
    
    This model is a probabilistic approach to sequence classification using a
    deep Markov model (DMM) structure, which allows for capturing temporal 
    dependencies and uncertainties in sequential data.
    
    Parameters:
        z_dim (int): The size of the latent state space. Default is 32.
        emission_dim (int): The size of the emission's output space. Default is 32.
        transition_dim (int): The size of the transition's output space. Default is 32.
        rnn_dim (int): The size of the RNN's hidden layer. Default is 32.
        num_layers (int): The number of RNN layers. Default is 1.
        dropout (float): The dropout rate for regularization. Default is 0.2.
        variance (float): The initial variance of the probabilistic layers. Default is 0.1.
        activation (str): The type of activation function to use. Default is 'relu'.
        bias (bool): Whether to use bias in the RNN layers. Default is True.
        seq_len (int): The length of the input sequences. Default is 20.
        **kwargs: Additional keyword arguments for the Classifier base class.
    
    Attributes:
        n_classes_ (int): Number of classes for classification. Set during fitting.
        n_features_in_ (int): Number of features expected during fitting. Set during fitting.
        z_0 (torch.nn.Parameter): Initial latent state parameter.
        z_q_0 (torch.nn.Parameter): Initial latent state parameter for the guide.
        h_0 (torch.nn.Parameter): Initial hidden state for the RNN.
    """

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
        Constructor for DMMClassifier.
        
        See class documentation for more details on the parameters.
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
        Define the generative model for a deep Markov model (DMM) regressor.

        The model is a Pyro probabilistic model, which includes the definition of the prior over
        the latent variables z and the likelihood of the observations y given z. It is used for
        training the model with variational inference.

        Args:
            x (torch.Tensor): The observed input features with shape (batch_size, T_max, input_dim),
                            where T_max is the maximum sequence length in the batch.
            y (torch.Tensor): The target variable with shape (batch_size, T_max, output_dim),
                            where T_max is the maximum sequence length in the batch.
            annealing_factor (float, optional): Factor to anneal the KL-divergence term in the loss
                                                during training (default is 1.0).

        Notes:
            - The `model` function is part of the Pyro model-guide pair required for SVI.
            - It includes annealing of the KL-divergence to stabilize training in the early epochs.
            - The `pyro.plate` construct is used to denote independent batches during the sampling process.
            - Time dependencies are handled through Pyro's `markov` context manager.
        """

        # Determine the maximum number of time steps from the input shape.
        T_max = x.size(1)

        # Expand the initial latent state `z_0` to match the batch size.
        z_prev = self.z_0.expand(x.size(0), self.z_0.size(0))

        # Use the `pyro.plate` construct to handle batches of data for conditional independence.
        with pyro.plate("z_minibatch", len(x)):
            # Loop over each time step while considering the previous state (Markovian assumption).
            for t in pyro.markov(range(1, T_max + 1)):
                # Obtain the location and scale for the latent state `z` at time `t`.
                z_loc, z_scale = self.transition(z_prev, x[:, t - 1, :])

                # Apply annealing to the KL divergence term to potentially stabilize training.
                with pyro.poutine.scale(scale=annealing_factor):
                    # Sample the latent variable `z_t` using the Normal distribution parameterized by `z_loc` and `z_scale`.
                    z_t = pyro.sample(
                        f"z_{t}",
                        dist.Normal(z_loc, z_scale).to_event(1)
                    )

                # Using the current latent state `z_t`, predict the categorical probabilities for the observation at time `t`.
                probs_t = self.emitter_cls(z_t, x[:, t - 1, :])

                # Instruct Pyro to observe the actual target `y` using the predicted categorical distribution.
                pyro.sample(
                    f"obs_y_{t}",
                    dist.Categorical(probs=probs_t).to_event(1),
                    obs=y
                )

                # Update the previous latent state to the current one for the next time step.
                z_prev = z_t
            
    def guide(self, x, y = None, annealing_factor = 1.0):
        """
        Define the variational guide for the deep Markov model (DMM) regressor.

        The guide serves as an approximate posterior that is optimized to resemble the true posterior
        of the latent variables given the observations. It is a parametrized distribution from which
        we can sample z and is defined using the same plates as the model for the latent variables.

        Args:
            x (torch.Tensor): The observed input features with shape (batch_size, T_max, input_dim),
                            where T_max is the maximum sequence length in the batch.
            y (torch.Tensor, optional): The target variable is not used in the guide and included
                                        for API consistency (default is None).
            annealing_factor (float, optional): Factor to anneal the KL-divergence term in the loss
                                                during training (default is 1.0).

        Notes:
            - The `guide` function is part of the Pyro model-guide pair required for SVI.
            - It specifies the family of distributions used for approximation of the posterior.
            - Time dependencies are handled through Pyro's `markov` context manager.
            - The hidden state of the RNN is used to compute the parameters of the variational distribution.
        """
        
        # Determine the maximum number of time steps from the input shape.
        T_max = x.size(1)

        # Register the module with Pyro to enable optimization of its parameters.
        pyro.module("dmm", self)

        # Expand and reformat the initial hidden state of the RNN to match the batch size and layers.
        h_0_contig = self.h_0.expand(self.num_layers, x.size(0), self.rnn.hidden_size).contiguous()

        # Process the input `x` through the RNN to obtain the output for each time step.
        rnn_output, _ = self.rnn(x, h_0_contig)

        # Reverse the output of the RNN to match the reversed order of the guide relative to the model.
        rnn_output = rnn_output.flip(1)
        
        # Expand the initial approximate latent state `z_q_0` to match the batch size.
        z_prev = self.z_q_0.expand(x.size(0), self.z_dim)
        
        # Use the `pyro.plate` construct to handle batches of data for conditional independence.
        with pyro.plate("z_minibatch", len(x)):
            # Loop over each time step while considering the previous state (Markovian assumption).
            for t in pyro.markov(range(1, T_max + 1)):
                # Obtain the location and scale for the latent state `z` at time `t` using the RNN output.
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                # Define the Normal distribution for the latent variable `z` at time `t`.
                z_dist = dist.Normal(z_loc, z_scale)

                # Temporarily adjust the scale of the score function, here it's effectively not adjusted (None).
                with pyro.poutine.scale(None, annealing_factor):
                    # Sample the latent variable `z_t` from the defined distribution `z_dist`.
                    z_t = pyro.sample(f"z_{t}", z_dist.to_event(1))

                # Update the previous latent state to the current one for the next time step.
                z_prev = z_t

    def forward(self, x):
        """
        Perform the forward pass with the neural network, predicting class logits at each time step of the input sequence.
        
        The forward pass uses the latent variables sampled by the guide during the variational inference process
        and passes them to the emitter to produce logits for class probabilities at each time step.
        The final prediction is the average of these logits across all time steps.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, num_features),
                            where sequence_length is the length of the time series, and num_features
                            is the number of features at each time step.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes), which contains the averaged
                        logits for class probabilities over the sequence for each instance in the batch.
                        
                        The logits can then be passed through a softmax function to obtain probability
                        distributions over the classes for each instance.

        Example:
            >>> x = torch.rand(32, 20, 10) # A batch of 32 sequences, each of length 20, with 10 features each
            >>> model = DMMClassifier(...)
            >>> logits = model.forward(x)
            >>> probabilities = torch.nn.functional.softmax(logits, dim=1)
            >>> predicted_classes = torch.argmax(probabilities, dim=1)
            # predicted_classes now contains the most likely class for each instance in the batch

        Note:
            The guide function must be defined and properly initialized in the class, as it is used to sample the
            latent variables and the trace of the guide is used within this forward pass.

            The emitter must also be defined and should represent a neural network module capable of
            taking the latent variables `z_t` and input features `x` to produce the logits for class probabilities.
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