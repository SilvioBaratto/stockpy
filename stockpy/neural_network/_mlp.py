import torch
import torch.nn as nn
import torch.nn.functional as F
from stockpy.base import Regressor
from stockpy.base import Classifier 
from stockpy.utils import get_activation_function

__all__ =  ['MLPClassifier', 'MLPRegressor']

class MLP(nn.Module):

    class MLP:
        """
        Multilayer Perceptron (MLP) class.

        This class defines a basic feedforward neural network with configurable
        number of hidden layers, dropout rate, activation function, and bias term.

        Parameters
        ----------
        hidden_size : int or list of int, optional
            The size of each hidden layer. If an int is provided, it is treated as
            the size for a single hidden layer. If a list is provided, each element
            corresponds to the size of a layer in the MLP. By default, 32.
        dropout : float, optional
            The dropout rate for regularization. Values should be between 0 and 1.
            By default, 0.2.
        activation : str, optional
            The type of activation function to use in the hidden layers. By default, 'relu'.
        bias : bool, optional
            If True, adds a learnable bias to the layers. By default, True.
        **kwargs
            Arbitrary keyword arguments which could be used by subclasses or in method calls.

        Attributes
        ----------
        hidden_sizes : list of int
            Stores the size of each hidden layer after processing the input argument.
        output_size : int
            The size of the output layer, which is determined by the task (classification or regression).
        criterion : torch.nn.modules.loss
            The loss function used for training the network, specific to the type of task.
        layers : torch.nn.Sequential
            The actual neural network layers stored in a sequential container.
        """

    def __init__(self,
                 hidden_size=32,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 batch_norm=False,
                 layer_norm=False,
                 **kwargs):

        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

    def initialize_module(self):
        """
        Initializes the layers and the criterion of the MLP based on its configuration.

        It checks the type of task (classification or regression) and sets up the output layer
        and loss criterion accordingly. It then constructs the hidden layers and the output layer,
        applying the specified activation function and dropout between each hidden layer.

        The initialization of the layers is based on the input size (number of features) and the
        specified architecture of the network (number of hidden layers and their sizes).

        """
        # Checks if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(self.hidden_size, int):
            self.hidden_sizes = [self.hidden_size]
        else:
            self.hidden_sizes = self.hidden_size

        if isinstance(self, Classifier):
            self.output_size = self.n_classes_
            self.criterion_ = nn.NLLLoss()
        elif isinstance(self, Regressor):
            self.output_size = self.n_outputs_
            self.criterion_ = nn.MSELoss()

        layers = []
        input_size = self.n_features_in_
        # Creates the layers of the neural network
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size, bias=self.bias))
            # append batch normalization layer if specified
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            # append layer normalization layer if specified
            if self.layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(get_activation_function(self.activation))
            layers.append(nn.Dropout(self.dropout))
            input_size = hidden_size

        # Appends the output layer to the neural network
        layers.append(nn.Linear(input_size, self.output_size))
        # Stacks all the layers into a sequence
        self.layers = nn.Sequential(*layers)

    @property
    def model_type(self):
        return "ffnn"

class MLPClassifier(Classifier, MLP):
    """
    MLPClassifier extends the MLP model with classification capabilities.

    This class is a type of neural network specifically designed for classification tasks. 
    It uses a softmax layer as the final layer to obtain probabilities for the classification classes.

    Parameters
    ----------
    hidden_size : int or list of int, optional
        The size of each hidden layer in the MLP. By default, a single hidden layer with 32 units.
    dropout : float, optional
        The dropout rate used for regularization to prevent overfitting. By default, 0.2.
    activation : str, optional
        The activation function for the hidden layers. By default, 'relu'.
    bias : bool, optional
        Indicates whether or not to use bias terms in the layers. By default, True.
    **kwargs
        Additional keyword arguments for the base Classifier class.

    Attributes
    ----------
    Inherits all attributes from the MLP and Classifier classes.

    Raises
    ------
    RuntimeError
        If the `forward` method is called before the model is fitted.

    Methods
    -------
    forward(x)
        Defines the forward pass of the classifier.
    """

    def __init__(self,
                 hidden_size=32,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 batch_norm=False,
                 layer_norm=False,
                 **kwargs):
        """
        Constructor for the MLPClassifier.

        Initializes a new instance of the MLPClassifier with the specified configuration.
        It configures the neural network layers and their respective parameters.
        """

        Classifier.__init__(self, **kwargs)
        MLP.__init__(self, 
                     hidden_size=hidden_size, 
                     dropout=dropout, 
                     activation=activation, 
                     bias=bias, 
                     batch_norm=batch_norm,
                     layer_norm=layer_norm,
                     **kwargs
                     )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the MLP classifier.

        Takes an input tensor `x`, passes it through the neural network layers,
        and applies the softmax function to the final layer to get the probabilities
        for each class.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the features.

        Returns
        -------
        torch.Tensor
            The output tensor after the softmax layer, representing the probability
            distribution over the target classes.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet (i.e., `fit` method not called).
        """
        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        # Returns the output of the forward pass of the neural network
        x = self.layers(x)
        x = F.softmax(x, dim=-1)
        return x
    
class MLPRegressor(Regressor, MLP):
    """
    MLPRegressor is a multi-layer perceptron for regression tasks.

    It predicts continuous values by using a neural network model without a softmax layer,
    unlike its classification counterpart.

    Parameters
    ----------
    hidden_size : int or list of int, optional
        The size of each hidden layer in the MLP. If it is an integer, it is the size of a single hidden layer;
        if it is a list, each element is the size of a layer. By default, it is a single layer with 32 units.
    dropout : float, optional
        The dropout rate for regularization to reduce overfitting. By default, 0.2.
    activation : str, optional
        The activation function to use for the hidden layers. By default, 'relu'.
    bias : bool, optional
        If True, layers will use bias terms. By default, True.
    **kwargs
        Additional keyword arguments inherited from the base Regressor class.

    Attributes
    ----------
    Inherits all attributes from the MLP and Regressor classes.

    Raises
    ------
    RuntimeError
        If the `forward` method is invoked before the model has been fitted.

    Methods
    -------
    forward(x)
        Defines the forward propagation of the regression model.
    """

    def __init__(self,
                 hidden_size=32,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 batch_norm=False,
                 layer_norm=False,
                 **kwargs):
        """
        Constructor for the MLPRegressor.

        Initializes a new instance of MLPRegressor with the specified configuration.
        It constructs the layers and their respective parameters for the neural network based on the given arguments.
        """

        Regressor.__init__(self, **kwargs)
        MLP.__init__(self, 
                     hidden_size=hidden_size, 
                     dropout=dropout, 
                     activation=activation, 
                     bias=bias, 
                     batch_norm=batch_norm,
                     layer_norm=layer_norm,
                     **kwargs
                     )
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the MLP regressor.

        Processes the input tensor `x` through the neural network layers to predict continuous values.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the features for regression.

        Returns
        -------
        torch.Tensor
            The output tensor from the final layer of the neural network, representing the predicted values.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet (i.e., `fit` method not called).
        """
        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        # Returns the output of the forward pass of the neural network
        return self.layers(x)  
