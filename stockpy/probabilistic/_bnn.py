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

    def reset_weights(self):
        """
        Reinitializes the model weights.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def initialize_module(self):
        """
        Initializes the layers of the neural network based on configuration.
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

    def __init__(self,
                 hidden_size=32,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 **kwargs):
        """
        Initializes the MLPClassifier object with given or default parameters.
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
        Defines the forward pass of the neural network.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
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
        Defines the guide (i.e. variational distribution) for the model.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
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

        preds = []

        for _ in range(self.n_outputs_):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(x)
            preds.append(guide_trace.nodes["_RETURN"]["value"])

        preds = torch.stack(preds)

        return preds.mean(0)
    
class BNNRegressor(Regressor, BNN):

    def __init__(self,
                 hidden_size=32,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 **kwargs):
        """
        Initializes the MLPClassifier object with given or default parameters.
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
        Defines the forward pass of the neural network.
        :param x: The input tensor.
        :returns: The output tensor, corresponding to the predicted target variable(s).
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
        Defines the guide (i.e. variational distribution) for the model.
        :param x: The input tensor.
        :param y: The output tensor, used only to match the model's signature.
        :returns: The output tensor, corresponding to the predicted target variable(s).
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

        preds = []

        for _ in range(self.n_outputs_):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(x)
            preds.append(guide_trace.nodes["_RETURN"]["value"])

        preds = torch.stack(preds)

        return preds.mean(0)