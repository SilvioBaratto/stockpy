import torch
import torch.nn as nn
import torch.nn.functional as F
from stockpy.base import Regressor
from stockpy.base import Classifier 
from stockpy.utils import get_activation_function

__all__ = ['BiGRUClassifier', 'BiGRURegressor']

class BiGRU(nn.Module):
    """
    Gated Recurrent Unit (BiGRU) based Recurrent Neural Network for sequence processing tasks.

    The BiGRU is capable of capturing dependencies in sequential data and is widely used in time-series analysis,
    natural language processing, and other domains where sequence data is prevalent.

    Parameters
    ----------
    rnn_size : int
        The number of features in the hidden state `h` of the BiGRU cells and the dimensionality of the output feature space.
    hidden_size : int or list of int
        The size of each hidden layer in the fully connected feedforward network that follows the BiGRU layers.
        If an int is provided, it specifies a single hidden layer size; a list specifies the size of each layer.
    num_layers : int
        The number of BiGRU layers to stack. More layers can capture more complex dependencies but also increase computational complexity.
    dropout : float
        Dropout rate applied to the BiGRU layers (if `num_layers` > 1) and the subsequent fully connected layers for regularization.
    activation : str
        The type of activation function to apply after each fully connected layer except the output layer. Common options are 'relu', 'tanh', etc.
    bias : bool
        Indicates whether or not to include bias terms in the BiGRU cells and fully connected layers.
    seq_len : int
        The length of the input sequences.
    **kwargs
        Arbitrary keyword arguments for additional configuration or for use by the `nn.Module` base class.

    Attributes
    ----------
    BiGRU : torch.nn.GRU
        The BiGRU layer that processes input sequences and outputs a sequence or a final hidden state.
    layers : torch.nn.Sequential
        A sequence of fully connected layers that follows the BiGRU layer for further processing or for generating the output.

    Methods
    -------
    initialize_module(self)
        Initializes the BiGRU and fully connected layers. Typically, this would include setting up the weights and biases for the layers.
    """

    def __init__(self,
                 rnn_size = 32,
                 hidden_size=32,
                 num_layers=1,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 batch_norm=False,
                 layer_norm=False,
                 **kwargs):
        """
        Constructor for the BiGRU class.

        Initializes a new instance of BiGRU with the specified configuration for sequence processing tasks. 
        It constructs an BiGRU layer followed by a series of fully connected layers based on the given arguments.
        """

        super().__init__()

        self.rnn_size = rnn_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.seq_len = seq_len
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

    def initialize_module(self):
        """
        Constructs and initializes the BiGRU layers and fully connected layers.

        This setup is based on the initial configuration provided to the model constructor. It creates a BiGRU layer
        to process the input sequences followed by a series of fully connected layers for further processing
        or output generation.

        The method automatically determines the output size based on whether the model instance is for classification
        or regression, derived from `Classifier` or `Regressor`.

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

        self.bigru = nn.GRU(input_size=self.n_features_in_,
                             hidden_size=self.rnn_size,
                             num_layers=self.num_layers,
                             bidirectional=True,
                             batch_first=True,
                             bias=self.bias)
        
        layers = []

        fc_input_size = self.rnn_size * 2
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(fc_input_size, hidden_size, bias=self.bias))
            # append batch normalization layer if specified
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            # append layer normalization layer if specified
            if self.layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(get_activation_function(self.activation))
            layers.append(nn.Dropout(self.dropout))

            fc_input_size = hidden_size

        # Appends the output layer to the neural network
        layers.append(nn.Linear(fc_input_size, self.output_size)) 

        self.layers = nn.Sequential(*layers)

    @property
    def model_type(self):
        return "rnn"
    
class BiGRUClassifier(Classifier, BiGRU):
    """
    A classifier that uses a Gated Recurrent Unit (BiGRU) network for sequence classification tasks.

    The `BiGRUClassifier` extends both `Classifier` for generic classification functionalities and `BiGRU` for handling
    sequential data. It is tailored for sequence classification, making it suitable for tasks such as time series 
    classification, text categorization, and more.

    Parameters
    ----------
    rnn_size : int
        The number of features in the hidden state `h` of each BiGRU layer.
    hidden_size : int or list of int
        The number of features in the hidden layer(s) of the classifier. Can be a list to specify the size of each layer.
    num_layers : int
        The number of stacked BiGRU layers.
    dropout : float
        The dropout probability for the dropout layers in the classifier.
    activation : str
        The activation function for the hidden layers.
    bias : bool
        Whether to use bias terms in the BiGRU and linear layers.
    seq_len : int
        The length of the input sequences.

    Methods
    -------
    __init__(...)
        Constructor for `BiGRUClassifier` which initializes the base BiGRU structure and classification specific layers.
    forward(x)
        Defines the forward pass of the model.

    Raises
    ------
    RuntimeError
        If the forward pass is called before the model is properly configured.

    Notes
    -----
    The rest of the methods from `Classifier` and `BiGRU` are inherited.
    """

    def __init__(self,
                 rnn_size = 32,
                 hidden_size=32,
                 num_layers=1,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 batch_norm=False,
                 layer_norm=False,
                 **kwargs):
        """
        Initializes the MLPClassifier object with given or default parameters.
        """
        Classifier.__init__(self, classes=None, **kwargs)
        BiGRU.__init__(self,
                        rnn_size=rnn_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        activation=activation,
                        seq_len=seq_len,
                        bias=bias,
                        batch_norm=batch_norm,
                        layer_norm=layer_norm,
                        **kwargs
                        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the classifier.

        The input data `x` is passed through the BiGRU layers, and the output of the last BiGRU layer is then
        fed into the fully connected layers. The final output is passed through a softmax layer to obtain 
        the classification probabilities.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the features of shape (batch_size, seq_len, features).

        Returns
        -------
        torch.Tensor
            The output tensor containing the class probabilities of shape (batch_size, num_classes).

        Raises
        ------
        RuntimeError
            If the input tensor `x` does not have the correct shape or type.
        """

        # Initializing the initial hidden state for BiGRU layers with zeros
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.rnn_size, requires_grad=True)

        # Passing the input and initial hidden state through the BiGRU layers
        # The BiGRU returns the output for each time step as well as the last hidden state
        out, _ = self.bigru(x, h_0)

        # Taking the output of the last time step after the final BiGRU layer
        # and passing it through the fully connected layer stack
        out = self.layers(out[:, -1, :])  # (batch_size, rnn_size) -> (batch_size, hidden_sizes[-1])

        # Applying softmax to the output layer to get the probability distribution over classes
        out = F.softmax(out, dim=-1)  # (batch_size, hidden_sizes[-1]) -> (batch_size, num_classes)

        return out

class BiGRURegressor(Regressor, BiGRU):
    """
    A regressor that uses a Gated Recurrent Unit (BiGRU) network for sequence regressions tasks.

    The `BiGRURegressor` extends both `Regressor` and `BiGRU` classes, leveraging the BiGRU capabilities for sequence 
    processing and applying it to regression problems.

    Parameters
    ----------
    rnn_size : int
        The number of features in the hidden state `h` of each BiGRU layer.
    hidden_size : int or list of int
        The number of features in the hidden layer(s) of the regressor. Can be a list to specify the size of each layer.
    num_layers : int
        The number of stacked BiGRU layers.
    dropout : float
        The dropout probability for the dropout layers in the regressor.
    activation : str
        The activation function for the hidden layers.
    bias : bool
        Whether to use bias terms in the BiGRU and linear layers.
    seq_len : int
        The length of the input sequences.

    Methods
    -------
    __init__(...)
        Constructor for `BiGRURegressor` which initializes the base BiGRU structure and regression specific layers.
    forward(x)
        Defines the forward pass of the model.

    Raises
    ------
    RuntimeError
        If the forward pass is called before the model is properly configured.

    Notes
    -----
    The rest of the methods from `Classifier` and `BiGRU` are inherited.
    """

    def __init__(self,
                 rnn_size = 32,
                 hidden_size=32,
                 num_layers=1,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 batch_norm=False,
                 layer_norm=False,
                 **kwargs):
        """
        Initializes the MLPClassifier object with given or default parameters.
        """
        Regressor.__init__(self, **kwargs)
        BiGRU.__init__(self, 
                     rnn_size=rnn_size,
                     hidden_size=hidden_size, 
                     num_layers=num_layers,
                     dropout=dropout, 
                     activation=activation, 
                     seq_len=seq_len,
                     bias=bias, 
                     batch_norm=batch_norm,
                     layer_norm=layer_norm,
                     **kwargs
                     )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BiGRURegressor model.

        The method processes the input sequence `x` through the BiGRU layers and then 
        through the fully connected layers to produce the regression output.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing the sequence of data. It should have dimensions 
            (batch_size, seq_len, n_features).

        Returns
        -------
        torch.Tensor
            The output tensor after processing the input through the BiGRU and linear layers. 
            For regression, this will typically have dimensions (batch_size, output_size), 
            where `output_size` corresponds to the predicted values for each sequence in the batch.
        """
        
        # Initialize the hidden state for the BiGRU
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.rnn_size, requires_grad=True)
        
        # Pass the input through the GRU layer
        out, _ = self.bigru(x, h_0)
        
        # Pass the final hidden state through the fully connected layers
        out = self.layers(out[:, -1, :])
    
        return out