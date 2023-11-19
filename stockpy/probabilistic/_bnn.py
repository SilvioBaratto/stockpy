import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from stockpy.base import Regressor
from stockpy.base import Classifier 
from stockpy.utils import get_activation_function

__all__ = ['BNNClassifier', 'BNNRegressor']

class BNN(PyroModule):
    """
    Implements a Bayesian Neural Network (BNN) using Pyro's probabilistic layers.

    Parameters
    ----------
    hidden_size : int or list of int
        Number of units in each hidden layer. An integer specifies a single layer neural network,
        while a list specifies the size of units in each layer of a multi-layer network.
    dropout : float, optional
        Dropout rate for regularization. Default value is 0.2.
    activation : str, optional
        The activation function to use. It can be 'relu', 'tanh', etc. Default is 'relu'.
    bias : bool, optional
        Whether to use bias in the layers. Default is True.
    **kwargs
        Arbitrary keyword arguments.

    Attributes
    ----------
    hidden_size : int or list of int
        Size of the hidden layers as provided in the parameter.
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    bias : bool
        Indicates if bias parameters are included in the network layers.

    """

    def __init__(self,
                 hidden_size=32,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 **kwargs):
        """
        Initializes the BNN object with given or default parameters.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.bias = bias

    def initialize_module(self):
        """
        Sets up the neural network layers with Bayesian priors on weights and biases.

        This method builds the neural network by adding sequences of layers with activation
        and dropout, concluding with an output layer. Weights and biases are treated as random
        variables with normal priors, embodying the Bayesian framework in this network.

        Raises
        ------
        AttributeError
            If attributes specifying network details (like output size) have not been set before
            initialization. This function presupposes the network's detailed setup as a classifier
            or regressor has been completed.

        """

        # Checks if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(self.hidden_size, int):
            self.hidden_sizes = [self.hidden_size]
        else:
            self.hidden_sizes = self.hidden_size

        if isinstance(self, Classifier):
            self.output_size = self.n_classes_
        elif isinstance(self, Regressor):
            self.output_size = self.n_outputs_

        layers = []
        input_size = self.n_features_in_
        # Creates the layers of the neural network
        for hidden_size in self.hidden_sizes:
            linear_layer = PyroModule[nn.Linear](input_size, hidden_size, bias=self.bias)

            linear_layer.weight = PyroSample(
                dist.Normal(0., 1.).expand([hidden_size, input_size]).to_event(2)
            )
            linear_layer.bias = PyroSample(
                dist.Normal(0., 1.).expand([hidden_size]).to_event(1)
            )

            layers.append(linear_layer)
            layers.append(get_activation_function(self.activation))
            layers.append(PyroModule[nn.Dropout](self.dropout))
            input_size = hidden_size

        # Appends the output layer to the neural network
        output_layer = PyroModule[nn.Linear](input_size, self.output_size)
        # Set prior on weights and biases for output layer
        output_layer.weight = PyroSample(
            dist.Normal(0., 1.).expand([self.output_size, input_size]).to_event(2)
        )
        output_layer.bias = PyroSample(
            dist.Normal(0., 1.).expand([self.output_size]).to_event(1)
        )

        layers.append(output_layer)
        # Stacks all the layers into a sequence
        self.layers = PyroModule[nn.Sequential](*layers)

    @property
    def model_type(self):
        return "ffnn"

class BNNClassifier(Classifier, BNN):
    """
    Implements a Bayesian Neural Network for classification tasks using Pyro's probabilistic layers.

    This class provides a classifier with Bayesian inference abilities using the principles outlined in the BNN class,
    bringing the uncertainty quantification of Bayesian methods to classification problems.

    Attributes
    ----------
    hidden_size : int or list of int
        Specifies the size(s) of the hidden layer(s). A single integer denotes a single-layer size, while a list
        defines multiple layers.

    dropout : float
        Dropout rate used for regularization to prevent overfitting.

    activation : str
        The type of activation function used in the network layers, such as 'relu' or 'tanh'.

    bias : bool
        Indicates whether the network layers should include a bias term.

    n_classes_ : int
        Number of unique classes in the classification task. Inherited from Classifier.

    n_features_in_ : int
        The number of input features the model expects. Inherited from Classifier.

    Parameters
    ----------
    hidden_size : int or list of int, optional
        The configuration for the number of neurons in the hidden layers.

    dropout : float, optional
        Specifies the dropout probability for regularization.

    activation : str, optional
        Designates the type of activation function to apply.

    bias : bool, optional
        Flag to include bias parameters in the layers.

    **kwargs
        Variable keyword arguments that can be passed to the BNN and Classifier base classes.
    """

    def __init__(self,
                 hidden_size=32,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 **kwargs):
        """
        Initializes the BNNClassifier with the specified parameters.

        Calls the initializer of the Classifier class to set up target class-related attributes and the 
        BNN initializer to set up Bayesian layers with appropriate priors for classification.
        """

        Classifier.__init__(self, **kwargs)
        BNN.__init__(self, 
                     hidden_size=hidden_size, 
                     dropout=dropout, 
                     activation=activation, 
                     bias=bias, 
                     **kwargs
                     )
            
    def model(self, x, y):
        """
        Defines the generative model for the Bayesian Neural Network classifier.

        This method specifies a prior distribution over the neural network's weights and biases and then defines the
        likelihood of the observed data, given these prior beliefs. The neural network's outputs are used to parameterize
        the likelihood function, which is assumed to be a Categorical distribution over the class labels.

        Parameters
        ----------
        x : torch.Tensor
            Input features tensor, typically of shape (n_samples, n_features), where n_samples is the number of samples
            and n_features is the number of features.

        y : torch.Tensor
            Target labels tensor, typically of shape (n_samples,), containing class labels for each sample.

        Raises
        ------
        RuntimeError
            If the network layers are not initialized prior to invocation, an error is raised indicating that the model
            cannot be defined without pre-initialized layers.

        Returns
        -------
        None
            The function has no return value. It contributes to the internal state of the Pyro model by sampling from
            distributions.
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
    
    def guide(self, x, y):
        """
        Specifies the variational guide for the Bayesian Neural Network classifier.

        The guide is a variational approximation to the posterior distribution over the network's weights and biases. 
        It is implemented as a mean-field approximation where each weight and bias has its own variational parameters
        that are learned during the inference process.

        Parameters
        ----------
        x : torch.Tensor
            Input features tensor, typically of shape (n_samples, n_features), where n_samples is the number of samples
            and n_features is the number of features. The input is used to determine the shape of the guide's parameters.

        y : torch.Tensor, optional
            Target labels tensor, which is not used in the guide but included for compatibility with the model signature.
            By default, it is None.

        Raises
        ------
        RuntimeError
            If the guide's parameters are not initialized before invocation, an error is raised to signal that the
            variational parameters cannot be defined without pre-initialization.

        Returns
        -------
        None
            The function has no return value. It defines the variational distribution's parameters within the Pyro
            framework.
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
        
    def forward(self, x):
        """
        Computes the predictive distribution of the Bayesian Neural Network by averaging 
        over multiple stochastic forward passes.

        This forward pass effectively uses Monte Carlo sampling to approximate the 
        network's predictions. The guide function is repeatedly sampled to obtain a set 
        of predictions which are then averaged to estimate the overall predictive mean.

        Parameters
        ----------
        x : torch.Tensor
            The input features tensor of shape (n_samples, n_features), where 
            n_samples is the number of samples and n_features is the number of 
            features in the input data.

        Returns
        -------
        torch.Tensor
            The tensor of averaged predictions, representing the mean of the 
            predictive distribution obtained from the Monte Carlo samples of the 
            guide function.
        """

        preds = []

        for _ in range(self.n_outputs_):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(x)
            preds.append(guide_trace.nodes["_RETURN"]["value"])

        preds = torch.stack(preds)

        return preds.mean(0)
    
class BNNRegressor(Regressor, BNN):
    """
    A Bayesian Neural Network regressor that uses Pyro's variational inference tools to perform
    regression with uncertainty estimation.

    Inherits from both Regressor for general regression functionalities and BNN for Bayesian
    network specifics.

    Parameters
    ----------
    hidden_size : int or list of int
        Specifies the number of neurons in each hidden layer. A single integer denotes a single
        hidden layer, whereas a list of integers specifies the size of each layer in a
        multi-layer perceptron architecture.
    dropout : float, default=0.2
        The dropout rate, specifying the probability that an element is set to zero to prevent
        overfitting during training.
    activation : str, default='relu'
        The type of activation function to use. Should be a string corresponding to a valid
        Pyro or PyTorch activation function (e.g., 'relu', 'tanh', 'sigmoid').
    bias : bool, default=True
        Indicates whether or not to use bias terms in the neural network layers.
    **kwargs
        Additional keyword arguments that are passed to the base Regressor and BNN classes
        to allow for more customization.

    """

    def __init__(self,
                 hidden_size=32,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 **kwargs):
        """
        Initialize the BNNRegressor object with the specified neural network architecture.

        Parameters
        ----------
        hidden_size : int or list of ints
            Specifies the size of the hidden layers. An integer denotes the size for a single layer,
            while a list denotes the size for each layer in a multi-layer network.
        dropout : float, optional
            The dropout rate for regularization, applied to all hidden layers. Defaults to 0.2.
        activation : str, optional
            The name of the activation function to use after each hidden layer. Defaults to 'relu'.
        bias : bool, optional
            Indicates whether to include bias parameters in the neural network layers. Defaults to True.
        **kwargs
            Arbitrary keyword arguments passed to the `Regressor` base class for further configuration.

        Raises
        ------
        ValueError
            If the `hidden_size` is not specified or the `activation` function is not recognized.

        """

        Regressor.__init__(self, **kwargs)
        BNN.__init__(self, 
                     hidden_size=hidden_size, 
                     dropout=dropout, 
                     activation=activation, 
                     bias=bias, 
                     **kwargs
                     )

    def model(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Defines the generative model for a Bayesian neural network where the weights
        and biases of the network are treated as random variables with priors.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the features.
        y : torch.Tensor
            The output tensor containing the response variables.

        Raises
        ------
        RuntimeError
            If the model is called before it is fit with training data.

        Notes
        -----
        This method is called during training and should not be used for making predictions.

        Returns
        -------
        torch.Tensor
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
        Defines the variational guide (approximate posterior) for the Bayesian neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the features.
        y : torch.Tensor, optional
            Unused parameter. It's included to match the signature of the model.

        Raises
        ------
        RuntimeError
            If the guide is called before the model is fit with training data.

        Notes
        -----
        This method defines the family of distributions used to approximate the posterior distribution
        of the weights and biases. It is typically parameterized by variational parameters that are learned
        during training.

        Returns
        -------
        torch.Tensor
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