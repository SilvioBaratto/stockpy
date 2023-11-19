import torch
import torch.nn as nn
import torch.nn.functional as F

from stockpy.base import Regressor
from stockpy.base import Classifier 
from stockpy.utils import get_activation_function

__all__ = ['BiLSTMClassifier', 'BiLSTMRegressor']

class BiLSTM(nn.Module):
    """
    LSTM is a recurrent neural network architecture that uses LSTM (Long Short-Term Memory) cells to 
    process sequences of data.

    It's designed to remember long-range dependencies and is often used in 
    time-series prediction, natural language processing, and other sequence-related tasks.

    Parameters
    ----------
    rnn_size : int
        The number of features in the hidden state h of the LSTM. It's also the output feature dimension 
        after processing the input sequence.
    hidden_size : int or list of int
        The size of each hidden layer in the fully connected layers after the LSTM layer. If it is an integer, 
        it is the size of a single hidden layer; if it is a list, each element is the size of a layer.
    num_layers : int
        The number of recurrent layers (i.e., number of LSTM layers stacked on each other).
    dropout : float
        The dropout rate used for regularization in both LSTM (if `num_layers` > 1) and fully connected layers.
    activation : str
        The activation function to use after each fully connected layer except for the output layer.
    bias : bool
        If True, layers will use bias terms.
    seq_len : int
        The length of the input sequence.
    **kwargs
        Additional keyword arguments that might be required for the base class `nn.Module`.

    Attributes
    ----------
    lstm : torch.nn.LSTM
        The LSTM layer of the network.
    layers : torch.nn.Sequential
        The Sequential container of fully connected layers following the LSTM layer.

    Methods
    -------
    initialize_module()
        Initializes the LSTM layer and fully connected layers of the neural network.
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
        Constructor for the LSTM class.

        Initializes a new instance of LSTM with the specified configuration for sequence processing tasks. 
        It constructs an LSTM layer followed by a series of fully connected layers based on the given arguments.
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
        Implements the initialization of the LSTM and fully connected layers.

        Sets up the architecture of the neural network based on the configuration provided in the constructor. 
        This includes initializing the LSTM layer with the defined `rnn_size` and `num_layers`, and the fully 
        connected layers according to `hidden_size` and `activation` function.

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

        self.bilstm = nn.LSTM(input_size=self.n_features_in_,
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

class BiLSTMClassifier(Classifier, BiLSTM):
    """
    LSTMClassifier is a neural network module for sequence classification tasks
    that uses an LSTM (Long Short-Term Memory) layer followed by a fully
    connected layer. It is suitable for tasks where the input data is a sequence
    and the output is a discrete class.

    Parameters
    ----------
    rnn_size : int
        The number of units in the LSTM layer.
    hidden_size : int
        The number of units in the hidden layer(s) following the LSTM layer.
    num_layers : int
        The number of layers in the LSTM.
    dropout : float
        If non-zero, introduces a dropout layer on the outputs of each LSTM layer
        except the last layer, with dropout probability equal to `dropout`.
    activation : str
        The activation function to use on the outputs of the hidden layers.
    bias : bool
        If `False`, then the layer does not use bias weights b_ih and b_hh.
        Default: `True`.
    seq_len : int
        The length of the input sequences.
    **kwargs : dict, optional
        Additional arguments passed to the `Classifier` base class.

    Attributes
    ----------
    criterion : torch.nn.Module
        The criterion that is used to compute the loss of the model.

    Methods
    -------
    forward(x)
        Defines the forward pass of the LSTM classifier.
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
        Constructs an LSTMClassifier instance with specified parameters for the LSTM
        and fully connected layers. It initializes base Classifier attributes and
        sets up the criterion as Negative Log-Likelihood Loss (NLLLoss).
        """

        Classifier.__init__(self, classes=None, **kwargs)
        BiLSTM.__init__(self,
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
        Forward pass through the LSTM and fully connected layers.

        This method takes an input sequence, processes it through the LSTM layer(s), and passes the final
        hidden state through the fully connected layer(s) to produce the output.

        Parameters
        ----------
        x : torch.Tensor
            The input data tensor for sequence classification. Expected to have
            dimensions (batch_size, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            The output tensor after processing through LSTM and fully connected layers.
            It contains the log probabilities of the classes for each sequence in the batch.

        Raises
        ------
        RuntimeError
            If the input tensor does not match the expected dimensions or if an operation
            within the forward pass fails.
        """
        # Ensures LSTM initial states h_0 and c_0 are reset to zeros at each forward call
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.rnn_size, requires_grad=True)
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.rnn_size, requires_grad=True)

        # Processes input through the LSTM layer
        out, (hn, cn) = self.bilstm(x, (h_0.detach(), c_0.detach()))

        # Takes the output of the last sequence step from LSTM layer
        out = self.layers(out[:, -1, :])

        # Applies softmax to output layer to obtain log probabilities for classification
        out = F.softmax(out, dim=-1)

        return out

class BiLSTMRegressor(Regressor, BiLSTM):
    """
    LSTMRegressor is a neural network module for sequence regression tasks that uses
    an LSTM (Long Short-Term Memory) layer followed by a fully connected layer. It is
    suitable for tasks where the input data is a sequence and the output is a continuous value.

    Parameters
    ----------
    rnn_size : int
        The number of units in the LSTM layer.
    hidden_size : int
        The number of units in the hidden layer(s) following the LSTM layer.
    num_layers : int
        The number of layers in the LSTM.
    dropout : float
        If non-zero, introduces a dropout layer on the outputs of each LSTM layer
        except the last layer, with dropout probability equal to `dropout`.
    activation : str
        The activation function to use on the outputs of the hidden layers.
    bias : bool
        If `False`, then the layer does not use bias weights b_ih and b_hh.
        Default: `True`.
    seq_len : int
        The length of the input sequences.
    **kwargs : dict, optional
        Additional arguments passed to the `Regressor` base class.

    Attributes
    ----------
    criterion : torch.nn.Module
        The criterion that is used to compute the loss of the model, which in the
        case of regression, is typically Mean Squared Error (MSE).

    Methods
    -------
    forward(x)
        Defines the forward pass of the LSTM regressor.
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
        Constructs an LSTMRegressor instance with specified parameters for the LSTM
        and fully connected layers. It initializes base Regressor attributes and
        sets up the criterion as Mean Squared Error Loss (MSELoss).
        """

        Regressor.__init__(self, **kwargs)
        BiLSTM.__init__(self, 
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
        Forward pass through the LSTM and fully connected layers.

        This method takes an input sequence, processes it through the LSTM layer(s), and passes the final
        hidden state through the fully connected layer(s) to produce the output.

        Parameters
        ----------
        x : torch.Tensor
            The input data tensor for sequence regression. Expected to have
            dimensions (batch_size, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            The output tensor after processing through LSTM and fully connected layers.
            It contains the continuous values predicted for each sequence in the batch.

        Raises
        ------
        RuntimeError
            If the input tensor does not match the expected dimensions or if an operation
            within the forward pass fails.
        """
        
        # Ensures LSTM initial states h_0 and c_0 are reset to zeros at each forward call
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.rnn_size, requires_grad=True)
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.rnn_size, requires_grad=True)

        # Processes input through the LSTM layer
        out, (hn, cn) = self.lstm(x, (h_0.detach(), c_0.detach()))

        # Takes the output of the last sequence step from LSTM layer
        out = self.layers(out[:, -1, :])

        # Returns the final output for regression
        return out