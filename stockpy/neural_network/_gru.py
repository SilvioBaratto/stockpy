import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from stockpy.base import Regressor
from stockpy.base import Classifier 
from stockpy.utils import get_activation_function

class GRU(nn.Module):

    """
    GRU is a recurrent neural network architecture that uses GRU (Gated Recurrent Unit) cells to 
    process sequences of data. It's designed to remember long-range dependencies and is often used in 
    time-series prediction, natural language processing, and other sequence-related tasks.

    Parameters:
        rnn_size : int
            The number of features in the hidden state h of the GRU. It's also the output feature dimension 
            after processing the input sequence.
        hidden_size : int or list of int
            The size of each hidden layer in the fully connected layers after the GRU layer. If it is an integer, 
            it is the size of a single hidden layer; if it is a list, each element is the size of a layer.
        num_layers : int
            The number of recurrent layers (i.e., number of GRU layers stacked on each other).
        dropout : float
            The dropout rate used for regularization in both GRU (if `num_layers` > 1) and fully connected layers.
        activation : str
            The activation function to use after each fully connected layer except for the output layer.
        bias : bool
            If True, layers will use bias terms.
        seq_len : int
            The length of the input sequence.
        **kwargs
            Additional keyword arguments that might be required for the base class `nn.Module`.

    Attributes:
        GRU : torch.nn.GRU
            The GRU layer of the network.
        layers : torch.nn.Sequential
            The Sequential container of fully connected layers following the GRU layer.

    Methods:
        initialize_module()
            Initializes the GRU layer and fully connected layers of the neural network.

    """

    def __init__(self,
                 rnn_size = 32,
                 hidden_size=32,
                 num_layers=1,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 **kwargs):
        """
        Constructor for the GRU class.

        Initializes a new instance of GRU with the specified configuration for sequence processing tasks. 
        It constructs an GRU layer followed by a series of fully connected layers based on the given arguments.
        """

        super().__init__()

        self.rnn_size = rnn_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.seq_len = seq_len

    def initialize_module(self):
        """
        Constructs the GRU layers and fully connected layers based on the initial configuration.
        This method prepares the model by setting up the necessary layers and parameters.

        Raises:
            AttributeError
                If the model is not configured with necessary attributes like `n_features_in_` before calling.
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

        self.gru = nn.GRU(input_size=self.n_features_in_,
                             hidden_size=self.rnn_size,
                             num_layers=self.num_layers,
                             bidirectional=False,
                             batch_first=True,
                             bias=self.bias)
        
        layers = []

        fc_input_size = self.rnn_size
        # Creates the layers of the neural network
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(fc_input_size, hidden_size, bias=self.bias))
            layers.append(get_activation_function(self.activation))
            layers.append(nn.Dropout(self.dropout))

            fc_input_size = hidden_size

        # Appends the output layer to the neural network
        layers.append(nn.Linear(fc_input_size, self.output_size)) 

        self.layers = nn.Sequential(*layers)

    @property
    def model_type(self):
        return "rnn"

class GRUClassifier(Classifier, GRU):
    """
    A classifier that uses a Gated Recurrent Unit (GRU) network. This class is suitable for sequence classification tasks.
    
    The `GRUClassifier` extends both `Classifier` and `GRU` classes, inheriting the GRU-related properties and methods
    for sequence processing, while also implementing the specifics for classification.

    Attributes (inherited)
    ----------------------
    rnn_size : int
        The number of features in the hidden state `h` of each GRU layer.
    hidden_size : int
        The number of features in the hidden layer(s) of the classifier.
    num_layers : int
        The number of stacked GRU layers.
    dropout : float
        The dropout probability for the dropout layers in the classifier.
    activation : str
        The activation function for the hidden layers.
    bias : bool
        Whether to use a bias term in the GRU and linear layers.
    seq_len : int
        The length of the input sequences.
    criterion : nn.Module
        The loss function used during the training, which is Negative Log Likelihood Loss (NLLLoss) for classification.

    Methods
    -------
    __init__(rnn_size=32, hidden_size=32, num_layers=1, dropout=0.2, activation='relu', bias=True, seq_len=20, **kwargs)
        Constructor for `GRUClassifier` which initializes the base GRU structure and classification specific layers.

    forward(x)
        Defines the forward pass of the model.

    Raises
    ------
    RuntimeError
        If the forward pass is called before the model is fitted.

    Examples
    --------
    >>> gru_classifier = GRUClassifier(rnn_size=128, hidden_size=[64, 32], num_layers=2)
    >>> output = gru_classifier(input_sequence)
    """

    def __init__(self,
                 rnn_size = 32,
                 hidden_size=32,
                 num_layers=1,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 **kwargs):
        """
        Initializes the MLPClassifier object with given or default parameters.
        """
        Classifier.__init__(self, **kwargs)
        GRU.__init__(self,
                        rnn_size=rnn_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        activation=activation,
                        seq_len=seq_len,
                        bias=bias,
                        **kwargs
                        )

        self.criterion = nn.NLLLoss()

    def forward(self, x):
        """
        Forward pass through the model.
        """
        
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        out, _ = self.gru(x, h_0)
        
        # Concat the final forward and backward hidden states
        out = self.layers(out[:, -1, :])
        
        out = F.softmax(out, dim=-1)

        return out

class GRURegressor(Regressor, GRU):

    def __init__(self,
                 rnn_size = 32,
                 hidden_size=32,
                 num_layers=1,
                 dropout=0.2,
                 activation='relu',
                 bias=True,
                 seq_len=20,
                 **kwargs):
        """
        Initializes the MLPClassifier object with given or default parameters.
        """
        Regressor.__init__(self, **kwargs)
        GRU.__init__(self, 
                     rnn_size=rnn_size,
                     hidden_size=hidden_size, 
                     num_layers=num_layers,
                     dropout=dropout, 
                     activation=activation, 
                     seq_len=seq_len,
                     bias=bias, 
                     **kwargs
                     )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass through the model.
        """
        
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        out, _ = self.gru(x, h_0)
        
        # Concat the final forward and backward hidden states
        out = self.layers(out[:, -1, :])
     
        return out