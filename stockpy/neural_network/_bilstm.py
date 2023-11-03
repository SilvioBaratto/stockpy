import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from stockpy.base import Regressor
from stockpy.base import Classifier 
from stockpy.utils import get_activation_function

class BiLSTM(nn.Module):

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

        self.bilstm = nn.LSTM(input_size=self.n_features_in_,
                             hidden_size=self.rnn_size,
                             num_layers=self.num_layers,
                             bidirectional=True,
                             batch_first=True,
                             bias=self.bias)
        
        layers = []

        fc_input_size = self.rnn_size * 2
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

class BiLSTMClassifier(Classifier, BiLSTM):

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
        Classifier.__init__(self, classes=None, **kwargs)
        BiLSTM.__init__(self,
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
        h_0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.rnn_size))

        c_0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.rnn_size))

        out, (hn, cn) = self.bilstm(x, (h_0, c_0))
        
        # Concat the final forward and backward hidden states
        out = self.layers(out[:, -1, :])
        
        out = F.softmax(out, dim=-1)

        return out

class BiLSTMRegressor(Regressor, BiLSTM):

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
        BiLSTM.__init__(self, 
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
        h_0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.rnn_size))

        c_0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.rnn_size))

        out, (hn, cn) = self.bilstm(x, (h_0, c_0))
        
        # Concat the final forward and backward hidden states
        out = self.layers(out[:, -1, :])

        return out