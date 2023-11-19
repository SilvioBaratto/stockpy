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

__all__ = ['BCNNClassifier', 'BCNNRegressor']

class BCNN(PyroModule):
    """
    A Bayesian Convolutional Neural Network (BCNN) class implementing a probabilistic CNN with Pyro.

    This class allows the instantiation of CNN layers with prior distributions on their parameters,
    making it suitable for Bayesian inference. It can be extended to both classification and regression
    tasks by specifying the output size according to the problem.

    Parameters
    ----------
    hidden_size : int or list of int
        The number of features in the hidden fully connected layers.
    num_filters : int
        The number of filters in the convolutional layers.
    kernel_size : int
        The size of the convolutional kernel.
    pool_size : int
        The size of the window for max pooling.
    dropout : float
        The dropout probability for dropout layers.
    activation : str
        The name of the activation function to be used in the network layers.
    bias : bool
        Whether to use biases in the layers or not.

    Methods
    -------
    initialize_module()
        Initializes the neural network layers and assigns priors to the parameters based on the configuration.

    Examples
    --------
    >>> bcnn = BCNN(hidden_size=64, num_filters=128, kernel_size=5)
    >>> bcnn.initialize_module()
    """

    def __init__(self,
                 hidden_size=32,
                 num_filters=32,
                 kernel_size=3,
                 pool_size=2,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 **kwargs):
        def __init__(self, hidden_size=64, num_filters=128, kernel_size=5, pool_size=2, dropout=0.5, activation='relu', bias=True, **kwargs):
            """
            Initializes the BCNN object with given or default parameters. Can accept additional
            keyword arguments to pass to the PyroModule base class.
            """

        super().__init__()

        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout = dropout
        self.activation = activation
        self.bias = bias

    def initialize_module(self):
        """
        Initializes the layers of the BCNN based on configuration provided at initialization.

        It defines the convolutional, max pooling, flattening layers, and fully connected layers
        with priors on their weights and biases. The network's architecture is determined by the
        specified attributes such as `hidden_size`, `num_filters`, `kernel_size`, `pool_size`, 
        and `dropout`. 

        This method should be called after the network's required output size is known and set.

        Raises
        ------
        AttributeError
            If `n_classes_` or `n_outputs_` is not set for classification or regression
            before calling this method.
        """

        # Checks if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(self, Classifier):
            self.output_size = self.n_classes_
        elif isinstance(self, Regressor):
            self.output_size = self.n_outputs_

        # Check if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(self.hidden_size, int):
            self.hidden_sizes = [self.hidden_size]
        else:
            self.hidden_sizes = self.hidden_size

        self.input_size = self.n_features_in_

        # Initializes a list to store the layers of the neural network
        layers = [PyroModule[nn.Conv1d](1, self.num_filters, self.kernel_size),  # 1D convolutional layer
                  get_activation_function(self.activation),
                  PyroModule[nn.MaxPool1d](self.pool_size),  # Max pooling layer
                  PyroModule[nn.Flatten]()]  # Flatten layer for transforming the output for use in FC layers
                
        # Calculates the input size for the first FC layer after flattening
        current_input_size = self.num_filters * ((self.input_size - self.kernel_size + 1) \
                                                    // self.pool_size)
        
        # Create fully connected layers
        for hidden_size in self.hidden_sizes:
            linear_layer = PyroModule[nn.Linear](current_input_size, hidden_size)

            linear_layer.weight = PyroSample(
                dist.Normal(0., 1.).expand([hidden_size, current_input_size]).to_event(2)
            )
            linear_layer.bias = PyroSample(
                dist.Normal(0., 1.).expand([hidden_size]).to_event(1)
            )

            layers.append(linear_layer)
            layers.append(get_activation_function(self.activation))
            layers.append(PyroModule[nn.Dropout](self.dropout))
            input_size = hidden_size

        # Create output layer
        output_layer = PyroModule[nn.Linear](current_input_size, self.output_size)
        # Set prior on weights and biases for output layer
        output_layer.weight = PyroSample(
            dist.Normal(0., 1.).expand([self.output_size, input_size]).to_event(2)
        )
        output_layer.bias = PyroSample(
            dist.Normal(0., 1.).expand([self.output_size]).to_event(1)
        )

        layers.append(output_layer)

        # Combine layers into a sequential model
        self.layers = PyroModule[nn.Sequential](*layers)

    @property
    def model_type(self):
        return "cnn"

class BCNNClassifier(Classifier, BCNN):
    """
    A classifier that employs a Bayesian Convolutional Neural Network (BCNN) architecture for 
    classification tasks. This class is derived from the base `Classifier` and `BCNN` classes,
    integrating the Bayesian approach of `BCNN` with the task-specific structure and methods of a classifier.

    The `BCNNClassifier` is suitable for datasets that benefit from convolutional features extraction
    such as image and time-series classification.

    Attributes
    ----------
    hidden_size : int or list of int
        The number of features in the hidden fully connected layers.
    num_filters : int
        The number of filters in the convolutional layers.
    kernel_size : int
        The size of the convolutional kernel.
    pool_size : int
        The size of the window for max pooling.
    dropout : float
        The dropout probability for dropout layers.
    activation : str
        The name of the activation function to be used in the network layers.
    bias : bool
        Whether to use biases in the layers or not.
    n_classes_ : int
        The number of classes in the classification task.
    criterion : nn.Module
        The loss function used during training.
    """

    def __init__(self,
                 hidden_size=32,
                 num_filters=32,
                 kernel_size=3,
                 pool_size=2,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 **kwargs):
        """
        Initializes the `BCNNClassifier` object with given or default parameters. It configures the
        BCNN for the task of classification. Additional keyword arguments are passed to the `Classifier`
        and `BCNN` base class initializers.

        """

        Classifier.__init__(self, **kwargs)
        BCNN.__init__(self, 
                     hidden_size=hidden_size, 
                     num_filters=num_filters,
                     kernel_size=kernel_size,
                     pool_size=pool_size,
                     dropout=dropout, 
                     activation=activation, 
                     bias=bias, 
                     **kwargs
                     )

    def model(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Defines the probabilistic model's forward pass using Pyro primitives. It is used
        during training to define the joint distribution over all random variables in the model.

        This method sets up the prior distributions on the weights and biases of the network's layers,
        and then samples from these priors to generate predictions. Observations are then sampled from
        a categorical distribution parameterized by the neural network outputs.

        Parameters
        ----------
        x : torch.Tensor
            The input data as a tensor.
        y : torch.Tensor
            The target labels as a tensor.

        Returns
        -------
        None
            This method doesn't return a value but registers Pyro samples.

        Raises
        ------
        RuntimeError
            If the model's layers have not been initialized before this method is called.
        """

        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PyroModule[nn.Linear]):
                # Set up prior for the weights and biases of each layer
                weight_prior = dist.Normal(torch.zeros_like(layer.weight), 
                                           torch.ones_like(layer.weight)).to_event(2)
                bias_prior = dist.Normal(torch.zeros_like(layer.bias), 
                                         torch.ones_like(layer.bias)).to_event(1)

                # Sample from the prior
                layer.weight = pyro.sample(f"weight_{i}", weight_prior)
                layer.bias = pyro.sample(f"bias_{i}", bias_prior)
        
        with pyro.plate("data", x.shape[0]):
            out = self.layers(x)
            obs = pyro.sample("obs", dist.Categorical(logits=out).to_event(1), obs=y)
    
    def guide(self, x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        """
        Defines the guide (the variational distribution) for the probabilistic model. The guide specifies
        the variational family and learns the parameters of this family during training.

        In this guide, normal distributions are set up for the weights and biases with learnable parameters,
        and these distributions are sampled during the variational inference process. This method
        is used in tandem with the model during training to optimize the variational parameters.

        Parameters
        ----------
        x : torch.Tensor
            The input data as a tensor.
        y : torch.Tensor, optional
            The target labels as a tensor. Defaults to None as it is not used in the guide.

        Returns
        -------
        torch.Tensor
            The predictions from the guide, obtained by passing the input data through the network.
        
        Raises
        ------
        RuntimeError
            If the model's layers have not been initialized before this method is called.
        """
        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PyroModule[nn.Linear]):
                weight_loc = pyro.param(f"weight_loc_{i}", torch.zeros_like(layer.weight))
                weight_scale_unconstrained = pyro.param(f"weight_scale_{i}", torch.ones_like(layer.weight))
                bias_loc = pyro.param(f"bias_loc_{i}", torch.zeros_like(layer.bias))
                bias_scale_unconstrained = pyro.param(f"bias_scale_{i}", torch.ones_like(layer.bias))

                # Apply softplus to ensure that scale is positive
                weight_scale = F.softplus(weight_scale_unconstrained)
                bias_scale = F.softplus(bias_scale_unconstrained)

                layer.weight = pyro.sample(f"weight_{i}", dist.Normal(weight_loc, weight_scale).to_event(2))
                layer.bias = pyro.sample(f"bias_{i}", dist.Normal(bias_loc, bias_scale).to_event(1))
            
        with pyro.plate("data", x.shape[0]):
            out = self.layers(x)
            preds = F.softmax(out, dim=-1)
            return preds
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates predictions by running the guide function multiple times and
        averaging the results.

        This method uses a Monte Carlo approach to approximate the predictive
        distribution of a Bayesian neural network. It runs the guide function,
        which acts as the variational distribution, multiple times and stacks the
        results to compute the mean prediction across all runs.

        Parameters
        ----------
        x : torch.Tensor
            The input data as a tensor.

        Returns
        -------
        torch.Tensor
            The averaged predictions as a tensor.

        Notes
        -----
        This method assumes that the `self.n_outputs_` attribute is set and
        reflects the number of times the guide should be run to generate
        predictions. Each guide run produces a sample from the variational
        posterior which are then averaged to form the final prediction.
        """

        preds = []

        for _ in range(self.n_outputs_):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(x)
            preds.append(guide_trace.nodes["_RETURN"]["value"])

        preds = torch.stack(preds)

        return preds.mean(0)

class BCNNRegressor(Regressor, BCNN):
    """
    Bayesian Convolutional Neural Network (BCNN) Regressor.

    This class implements a BCNN for regression tasks using Pyro's probabilistic models.
    Inherits from Regressor and BCNN classes.

    Attributes
    ----------
    hidden_size : int
        The number of nodes in each hidden layer.
    num_filters : int
        The number of convolutional filters.
    kernel_size : int
        The size of the convolutional kernel.
    pool_size : int
        The size of the pooling window.
    dropout : float
        Dropout rate for regularization.
    activation : str
        Type of activation function to use.
    bias : bool
        Whether to use bias in the convolutional layers.

    Parameters
    ----------
    hidden_size : int
        The number of nodes in the hidden layer(s).
    num_filters : int
        The number of filters in the convolutional layers.
    kernel_size : int
        The size of the kernel for convolutional layers.
    pool_size : int
        The size of the max pooling window.
    dropout : float
        The dropout rate for regularization during training.
    activation : str
        The activation function to use after convolutional layers.
    bias : bool
        If set to True, layers will use bias parameters.
    **kwargs
        Arbitrary keyword arguments passed to the parent classes.
    """

    def __init__(self,
                 hidden_size=32,
                 num_filters=32,
                 kernel_size=3,
                 pool_size=2,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 **kwargs):
        """
        Constructor for BCNNRegressor.

        Initializes the BCNN regressor with specified parameters. If a parameter is
        not provided, a default value is used.

        """

        Regressor.__init__(self, **kwargs)
        BCNN.__init__(self, 
                     hidden_size=hidden_size, 
                     num_filters=num_filters,
                     kernel_size=kernel_size,
                     pool_size=pool_size,
                     dropout=dropout, 
                     activation=activation, 
                     bias=bias, 
                     **kwargs
                     )

    def model(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Defines the probabilistic model for the BCNN regressor.

        This method sets up the priors for the neural network's parameters and 
        defines the likelihood for the observed data. It is part of the Pyro 
        model-guide pair for variational inference.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the features.
        y : torch.Tensor
            The target tensor containing the true output values.

        Raises
        ------
        RuntimeError
            If the method is called before the model is fitted.

        Notes
        -----
        The method's return type is None because it performs internal sampling 
        operations and is used by Pyro for setting up the probabilistic model.
        """
        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        
        # Returns the output of the forward pass of the neural network
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PyroModule[nn.Linear]):
                # Set up prior for the weights and biases of each layer
                weight_prior = dist.Normal(torch.zeros_like(layer.weight), torch.ones_like(layer.weight)).to_event(2)
                bias_prior = dist.Normal(torch.zeros_like(layer.bias), torch.ones_like(layer.bias)).to_event(1)

                # Sample from the prior
                layer.weight = pyro.sample(f"weight_{i}", weight_prior)
                layer.bias = pyro.sample(f"bias_{i}", bias_prior)

        # Observation model (likelihood)
        with pyro.plate("data", x.shape[0]):
            out = self.layers(x)
            # Sample from the likelihood
            obs = pyro.sample("obs", dist.Normal(out, 1.0).to_event(1), obs=y)

    
    def guide(self, x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        """
        Defines the variational guide (approximate posterior) for the BCNN regressor.

        This method sets up variational parameters (location and scale) for the
        neural network's parameters and samples from the variational posterior
        distribution. It complements the `model` method during variational inference.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the features.
        y : torch.Tensor, optional
            The target tensor containing the true output values.
            It is not used in the guide and is only present to match the signature of the model.

        Raises
        ------
        RuntimeError
            If the method is called before the model is fitted.

        Notes
        -----
        The method's return type is None as it is used by Pyro to set up the variational guide
        and does not directly return values during its execution.
        """
        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
                
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PyroModule[nn.Linear]):
                weight_loc = pyro.param(f"weight_loc_{i}", torch.zeros_like(layer.weight))
                weight_scale_unconstrained = pyro.param(f"weight_scale_{i}", torch.ones_like(layer.weight))
                bias_loc = pyro.param(f"bias_loc_{i}", torch.zeros_like(layer.bias))
                bias_scale_unconstrained = pyro.param(f"bias_scale_{i}", torch.ones_like(layer.bias))

                # Apply softplus to ensure that scale is positive
                weight_scale = F.softplus(weight_scale_unconstrained)
                bias_scale = F.softplus(bias_scale_unconstrained)

                layer.weight = pyro.sample(f"weight_{i}", dist.Normal(weight_loc, weight_scale).to_event(2))
                layer.bias = pyro.sample(f"bias_{i}", dist.Normal(bias_loc, bias_scale).to_event(1))

        with pyro.plate("data", x.shape[0]):
            out = self.layers(x)
            return out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates predictions by running the guide function multiple times and
        averaging the results.

        This method uses a Monte Carlo approach to approximate the predictive
        distribution of a Bayesian neural network. It runs the guide function,
        which acts as the variational distribution, multiple times and stacks the
        results to compute the mean prediction across all runs.

        Parameters
        ----------
        x : torch.Tensor
            The input data as a tensor.

        Returns
        -------
        torch.Tensor
            The averaged predictions as a tensor.

        Notes
        -----
        This method assumes that the `self.n_outputs_` attribute is set and
        reflects the number of times the guide should be run to generate
        predictions. Each guide run produces a sample from the variational
        posterior which are then averaged to form the final prediction.
        """

        preds = []

        for _ in range(self.n_outputs_):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(x)
            preds.append(guide_trace.nodes["_RETURN"]["value"])

        preds = torch.stack(preds)

        return preds.mean(0)