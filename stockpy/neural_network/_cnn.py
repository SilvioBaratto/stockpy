
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from stockpy.base import Regressor
from stockpy.base import Classifier 
from stockpy.utils import get_activation_function

class CNN(nn.Module):

    """
    A Convolutional Neural Network (CNN) module for processing sequential or spatial data.
    This module can be used for tasks such as image classification or time-series analysis,
    depending on the dimensionality (1D for sequences, 2D for images) specified during initialization.

    Attributes:
        hidden_size : int
            The number of units in the dense layer(s) after the convolutional layers.
        num_filters : int
            The number of filters in each convolutional layer.
        kernel_size : int
            The size of the kernel in the convolutional layers.
        pool_size : int
            The size of the window to take a max over in max pooling layers.
        dropout : float
            Dropout rate for regularization after the convolutional layers.
        activation : str
            The activation function to use after each convolutional layer.
        num_layers : int
            The number of convolutional layers in the network.
        num_channels : int
            The number of input channels (e.g., 1 for grayscale or 3 for RGB images).
        dim : int
            The dimensionality of the convolution (e.g., 1 for sequences, 2 for images).
        bias : bool
            Whether or not to include bias parameters in the convolutional layers.
        **kwargs : dict
            Additional arguments that are passed to the `nn.Module` initializer.

    Methods:
        initialize_module()
            Initializes the layers and configuration of the network based on the attributes.
            This includes convolutional layers, activation functions, pooling layers, dropout layers,
            and a final dense layer for output.

    """

    def __init__(self,
                 hidden_size=32,
                 num_filters=32,
                 kernel_size=3,
                 pool_size=2,
                 dropout=0.2,
                 activation='relu',
                 num_layers=1,
                 num_channels=1,
                 dim=1,
                 bias=True,
                 **kwargs):
        """
        Constructs the CNN with the specified parameters. This includes setting up
        the convolutional layers, pooling layers, and fully connected layers based on the
        provided sizes and dimensions.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.dim = dim

    def initialize_module(self):
        """
        Initializes the layers of the CNN based on the provided specifications.
        This method creates a sequence of convolutional layers, each followed by
        an activation function and a max pooling layer, and then a dropout layer if specified.
        Finally, it adds a flattening layer and a fully connected layer to produce the output.

        The method ensures that the convolutional layers are compatible with the input size,
        and that additional layers do not reduce the spatial dimensions to below zero. It also
        calculates the flattened size for the final dense layer after the convolutional and pooling layers.

        Raises:
            Warning
                If adding more layers would result in a negative dimension size.

        """
        if isinstance(self, Classifier):
            self.output_size = self.n_classes_
        elif isinstance(self, Regressor):
            self.output_size = self.n_outputs_

        # Check if hidden_sizes is a single integer and, if so, converts it to a list
        if isinstance(self.hidden_size, int):
            self.hidden_sizes = [self.hidden_size]
        else:
            self.hidden_sizes = self.hidden_size

        # Define the initial input size and layer list
        self.input_size = self.n_features_in_
        layers = []

        layers.extend([
            nn.Conv1d(1, self.num_filters, self.kernel_size, bias=self.bias),
            get_activation_function(self.activation),
            nn.MaxPool1d(self.pool_size),
        ])

        input_size = self.num_filters  # Update the number of channels for the next layer
        flattened_size = (self.input_size - self.kernel_size + 1) // self.pool_size

        for i in range(self.num_layers - 1):
            # Calculate what the new flattened_size would be if you added another layer
            new_flattened_size = (flattened_size - self.kernel_size + 1) // self.pool_size
            
            # Check if adding another layer would result in a negative flattened_size
            if new_flattened_size <= 0:
                warnings.warn("Cannot add more layers; doing so would result in negative dimension size.")
                break

            layers.extend([
                nn.Conv1d(input_size, self.num_filters, self.kernel_size, bias=self.bias),
                get_activation_function(self.activation),
                nn.MaxPool1d(self.pool_size),
            ])

            input_size = self.num_filters  # Update the number of channels for the next layer
            flattened_size = (flattened_size - self.kernel_size + 1) // self.pool_size  # Update flattened_size

            if i < self.num_layers - 1:
                layers.append(nn.Dropout(self.dropout))  # Add dropout layers in between

        # Add Flatten and Linear layers
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.num_filters * flattened_size, self.output_size))
        
        # Create the Sequential model
        self.layers = nn.Sequential(*layers)

    @property
    def model_type(self):
        return "cnn"

class CNNClassifier(Classifier, CNN):

    """
    CNNClassifier is a convolutional neural network for classification tasks, which extends the
    functionality of a generic Classifier and the CNN class. It is suitable for handling
    datasets that can benefit from spatial feature extraction, such as image or time-series data.

    Attributes inherited from CNN:
        hidden_size : int
            The number of units in the dense layer(s) after the convolutional layers.
        num_filters : int
            The number of filters in each convolutional layer.
        kernel_size : int
            The size of the kernel in the convolutional layers.
        pool_size : int
            The size of the window to take a max over in max pooling layers.
        dropout : float
            Dropout rate for regularization after the convolutional layers.
        activation : str
            The activation function to use after each convolutional layer.
        num_layers : int
            The number of convolutional layers in the network.
        num_channels : int
            The number of input channels (e.g., 1 for grayscale or 3 for RGB images).
        dim : int
            The dimensionality of the convolution (e.g., 1 for sequences, 2 for images).
        bias : bool
            Whether or not to include bias parameters in the convolutional layers.

    Attributes inherited from Classifier:
        criterion : torch.nn.modules.loss
            The loss function used for the classification task, set to negative log likelihood loss (NLLLoss).

    Methods inherited from CNN:
        initialize_module()
            Initializes the layers of the CNN with the specified configuration.

    Methods:
        __init__(hidden_size=32, num_filters=32, kernel_size=3, pool_size=2, dropout=0.2,
                activation='relu', num_layers=1, num_channels=1, dim=1, bias=True, **kwargs)
            Constructor for the CNNClassifier class.
        forward(x)
            Defines the forward pass of the classifier.
    """

    def __init__(self,
                 hidden_size=32,
                 num_filters=32,
                 kernel_size=3,
                 pool_size=2,
                 dropout=0.2,
                 activation='relu',
                 num_layers=1,
                 num_channels=1,
                 dim=1,
                 bias=True,
                 **kwargs):
        """
        Initializes the CNNClassifier with the specified parameters for the convolutional neural network.
        The parameters include settings for the convolutional and dense layers, as well as the overall
        architecture of the network.
        """

        Classifier.__init__(self, **kwargs)
        CNN.__init__(self, 
                     hidden_size=hidden_size, 
                     num_filters=num_filters,
                     kernel_size=kernel_size,
                     pool_size=pool_size,
                     dropout=dropout, 
                     activation=activation, 
                     num_layers=num_layers,
                     num_channels=num_channels,
                     dim=dim,
                     bias=bias, 
                     **kwargs
                     )

        self.criterion = nn.NLLLoss()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the CNNClassifier with the input tensor `x`. It applies
        the sequence of convolutional, activation, pooling, and fully connected layers defined
        in the model to produce a probability distribution over the target classes.

        Parameters:
            x : torch.Tensor
                A tensor representing the input data.

        Returns:
            torch.Tensor
                The output tensor after applying the softmax function, indicating the probability
                distribution over the target classes.

        Raises:
            RuntimeError
                If the model's layers are not defined (i.e., `initialize_module` has not been called).
        """

        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
        
        x = self.layers(x)

        # apply softmax
        x = F.softmax(x, dim=-1)

        return x
    
class CNNRegressor(Regressor, CNN):

    """
    CNNRegressor is a convolutional neural network designed for regression tasks. It inherits from
    the Regressor and CNN classes. This class is particularly suited to regression problems where
    the input data have a grid-like topology, such as time-series data or image pixels.

    Attributes inherited from CNN:
        hidden_size : int
            The number of units in the dense layer(s) after the convolutional layers.
        num_filters : int
            The number of filters in each convolutional layer.
        kernel_size : int
            The size of the kernel in the convolutional layers.
        pool_size : int
            The size of the window to take a max over in max pooling layers.
        dropout : float
            Dropout rate for regularization after the convolutional layers.
        activation : str
            The activation function to use after each convolutional layer.
        num_layers : int
            The number of convolutional layers in the network.
        num_channels : int
            The number of input channels (e.g., 1 for grayscale or 3 for RGB images).
        dim : int
            The dimensionality of the convolution (e.g., 1 for sequences, 2 for images).
        bias : bool
            Whether or not to include bias parameters in the convolutional layers.

    Attributes inherited from Regressor:
        criterion : torch.nn.modules.loss
            The loss function used for the regression task, set to mean squared error loss (MSELoss).

    Methods inherited from CNN:
        initialize_module()
            Initializes the layers of the CNN with the specified configuration.

    Methods:
        __init__(hidden_size=32, num_filters=32, kernel_size=3, pool_size=2, dropout=0.2,
                activation='relu', num_layers=1, num_channels=1, dim=1, bias=True, **kwargs)
            Constructor for the CNNRegressor class.
        forward(x)
            Defines the forward pass of the regressor.
    """

    def __init__(self,
                 hidden_size=32,
                 num_filters=32,
                 kernel_size=3,
                 pool_size=2,
                 dropout=0.2,
                 activation='relu',
                 num_layers=1,
                 num_channels=1,
                 dim=1,
                 bias=True,
                 **kwargs):
        """
        Initializes the CNNRegressor with the specified parameters for the convolutional neural network.
        The parameters include settings for the convolutional and dense layers, as well as the overall
        architecture of the network designed for regression.
        """

        Regressor.__init__(self, **kwargs)
        CNN.__init__(self, 
                     hidden_size=hidden_size, 
                     num_filters=num_filters,
                     kernel_size=kernel_size,
                     pool_size=pool_size,
                     dropout=dropout, 
                     activation=activation, 
                     num_layers=num_layers,
                     num_channels=num_channels,
                     dim=dim,
                     bias=bias, 
                     **kwargs
                     )

        self.criterion = nn.MSELoss()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the CNNRegressor with the input tensor `x`. It applies
        the sequence of convolutional, activation, pooling, and fully connected layers defined
        in the model to produce the output regression values.

        Parameters:
            x : torch.Tensor
                A tensor representing the input data.

        Returns:
            torch.Tensor
                The output tensor, corresponding to the predicted regression values.

        Raises:
            RuntimeError
                If the model's layers are not defined (i.e., `initialize_module` has not been called).
        """

        # Ensures the model has been fitted before making predictions
        if self.layers is None:
            raise RuntimeError("You must call fit before calling predict")
                
        x = self.layers(x)

        return x
