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

class BNN(PyroModule):

    """
    A Bayesian Neural Network (BNN) class that uses Pyro's probabilistic modeling capabilities.
    This class serves as a base for constructing neural networks with Bayesian inference.

    Attributes:
        hidden_size (int or list of int): Size of the hidden layers. If an integer is provided, 
            it is treated as the size for a single hidden layer. If a list is provided, each 
            element specifies the size of a layer in a multi-layer network.
        dropout (float): The dropout rate for regularization during training.
        activation (str): The type of activation function to use. Expected to be a valid 
            string that maps to a PyTorch activation function (e.g., 'relu', 'tanh').
        bias (bool): Indicates whether or not to include bias parameters in the network layers.

    Parameters:
        hidden_size (int or list of int): The number of neurons in the hidden layer(s).
        dropout (float): Probability of an element to be zeroed during training. Defaults to 0.2.
        activation (str): Type of activation function to use. Defaults to 'relu'.
        bias (bool): If set to True, layers will include a bias term. Defaults to True.
        **kwargs: Additional keyword arguments for more configurations.
    """

    def __init__(self,
                 hidden_size=32,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 **kwargs):
        """
        Initializes the MLP object with given or default parameters.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.bias = bias

    def initialize_module(self):
        """
        Initializes the neural network layers based on the attributes of the class instance.

        This method constructs the neural network architecture by creating a sequence of layers,
        each followed by an activation function and a dropout layer, ending with an output layer.
        The weights and biases of each layer are set to be samples from a normal distribution,
        forming the basis of the Bayesian approach in this neural network.

        Raises:
            AttributeError: If the network type-specific attributes, such as output size, are
                not set prior to calling this method. This method expects the network to be 
                properly configured as a classifier or regressor before initialization.

        Note:
            This method is typically called internally during the fitting process and should 
            not be invoked manually without proper configuration.
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
        to_device(self.layers, self.device)

    @property
    def model_type(self):
        return "ffnn"

class BNNClassifier(Classifier, BNN): 

    """
    BNNClassifier is a Bayesian Neural Network classifier that extends the Classifier and BNN base classes. 
    It applies Bayesian inference to neural network classification tasks. It combines the functionalities of 
    a standard classifier with the Bayesian layers and priors defined in the BNN class.

    Attributes inherited from BNN:
        hidden_size (int or list of int): Size of the hidden layers.
        dropout (float): The dropout rate for regularization during training.
        activation (str): The type of activation function to use.
        bias (bool): Indicates whether or not to include bias parameters in the network layers.

    Attributes inherited from Classifier:
        n_classes_ (int): The number of classes in the target labels.
        n_features_in_ (int): The number of features in the input data.

    Parameters:
        hidden_size (int or list of int): The number of neurons in the hidden layer(s).
        dropout (float): Probability of an element to be zeroed. Defaults to 0.2.
        activation (str): Type of activation function to use. Defaults to 'relu'.
        bias (bool): If set to True, layers will include a bias term. Defaults to True.
        **kwargs: Additional keyword arguments for the Classifier and BNN classes.
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

        self.criterion = nn.NLLLoss()
            
    def model(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Defines the generative model for a Bayesian neural network, specifying the prior and likelihood.
        The priors are defined over the weights and biases of the network's layers, and the observations
        are assumed to follow a Categorical distribution parameterized by the output of the network.

        Args:
            x (torch.Tensor): The input data tensor.
            y (torch.Tensor): The target tensor with class labels.

        Raises:
            RuntimeError: If the model's layers are not initialized before calling this method.

        Returns:
            None: This method does not return a value but instead samples the likelihood of the data.
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
        Defines the variational guide (approximate posterior) for a Bayesian neural network, specifying
        the variational family for the parameters of the network. This is a mean-field approximation where
        each parameter has its own variational parameter.

        Args:
            x (torch.Tensor): The input data tensor.
            y (torch.Tensor, optional): The target tensor. Default is None since it's not used in the guide.

        Raises:
            RuntimeError: If the model's layers are not initialized before calling this method.

        Returns:
            None: This method does not return a value but defines the variational distribution.
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
        Generates predictions by running the guide function multiple times and
        averaging the results.

        This method uses a Monte Carlo approach to approximate the predictive
        distribution of a Bayesian neural network. It runs the guide function,
        which acts as the variational distribution, multiple times and stacks the
        results to compute the mean prediction across all runs.

        Parameters:
            x (torch.Tensor): The input data as a tensor.

        Returns:
            torch.Tensor: The averaged predictions as a tensor.

        Notes:
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
    
class BNNRegressor(Regressor, BNN):

    """
    Bayesian Neural Network (BCNN) Regressor.

    This class implements a BNN for regression tasks using Pyro's probabilistic models.
    Inherits from Regressor and BNN classes.

    Parameters:
        hidden_size (int or list of int): The number of neurons in the hidden layer(s).
        dropout (float): Probability of an element to be zeroed. Defaults to 0.2.
        activation (str): Type of activation function to use. Defaults to 'relu'.
        bias (bool): If set to True, layers will include a bias term. Defaults to True.
        **kwargs: Additional keyword arguments for the Regressor and BNN classes.

    
    """

    def __init__(self,
                 hidden_size=32,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 **kwargs):
        """
        Initializes the BNNRegressor object by setting up the architecture of the Bayesian Neural Network
        for regression tasks with the given parameters.

        Args:
            hidden_size (int, list, or tuple): Size of hidden layers. If an integer is provided, it is treated as the size for a single hidden layer. If a list or tuple is provided, each element specifies the size for each hidden layer.
            dropout (float): The dropout rate for regularization. It is applied to all hidden layers.
            activation (str): The name of the activation function to use after each hidden layer.
            bias (bool): Indicates whether to include bias parameters in the neural network layers.
            **kwargs: Additional keyword arguments that are passed to the `Regressor` base class.

        Note:
            The `**kwargs` should contain all other necessary configurations required for the `Regressor` initialization, such as the number of features, number of outputs, and other specific settings.

        Raises:
            ValueError: If provided arguments are not valid or insufficient for initializing the network layers.
        """

        Regressor.__init__(self, **kwargs)
        BNN.__init__(self, 
                     hidden_size=hidden_size, 
                     dropout=dropout, 
                     activation=activation, 
                     bias=bias, 
                     **kwargs
                     )

        self.criterion = nn.MSELoss()
        
    def model(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Defines the generative model for a Bayesian neural network where the weights
        and biases of the network are treated as random variables with priors.

        Args:
            x (torch.Tensor): The input tensor containing the features.
            y (torch.Tensor): The output tensor containing the response variables.

        Raises:
            RuntimeError: If the model is called before it is fit with training data.

        Notes:
            This method is called during training and should not be used for making predictions.
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

        Args:
            x (torch.Tensor): The input tensor containing the features.
            y (torch.Tensor): Unused parameter. It's included to match the signature of the model.

        Raises:
            RuntimeError: If the guide is called before the model is fit with training data.

        Notes:
            This method defines the family of distributions used to approximate the posterior distribution
            of the weights and biases. It is typically parameterized by variational parameters that are learned
            during training.
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
        
    def forward(self, x):

        """
        Generates predictions by running the guide function multiple times and
        averaging the results.

        This method uses a Monte Carlo approach to approximate the predictive
        distribution of a Bayesian neural network. It runs the guide function,
        which acts as the variational distribution, multiple times and stacks the
        results to compute the mean prediction across all runs.

        Parameters:
            x (torch.Tensor): The input data as a tensor.

        Returns:
            torch.Tensor: The averaged predictions as a tensor.

        Notes:
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