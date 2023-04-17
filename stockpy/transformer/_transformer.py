import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_
from ..config import nn_args, shared, training

class Transformer(nn.Module):
    """
    A class representing a Transformer model for stock prediction.
    The Transformer model is designed to capture long-range dependencies in time series data, such as stock prices,
    by leveraging self-attention mechanisms. It consists of an embedding layer, followed by a series of
    Transformer Encoder layers and a fully connected layer for prediction.
    :param input_size: The number of input features for the Transformer model.
    :type input_size: int
    :param hidden_size: The number of dimensions for the model's internal representation.
    :type hidden_size: int
    :param num_layers: The number of Transformer Encoder layers in the model.
    :type num_layers: int
    :param output_size: The number of output units for the Transformer model, corresponding to the predicted target variable(s).
    :type output_size: int
    :param dropout: The dropout percentage applied between layers for regularization, preventing overfitting.
    :type dropout: float
    :example:
        >>> from stockpy.neural_network import Transformer
        >>> transformer = Transformer()
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(nn_args.input_size, nn_args.hidden_size)
        self.pos_encoding = PositionalEncoding(nn_args.hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                nn_args.hidden_size, nn_args.input_size, nn_args.hidden_size, shared.dropout),
            nn_args.num_layers
        )
        self.fc = nn.Linear(nn_args.hidden_size, nn_args.output_size)
        
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Transformer model.
        :param x: The input tensor.
        :type x: torch.Tensor
        :returns: The output tensor, corresponding to the predicted target variable(s).
        :rtype: torch.Tensor
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Average over sequence length
        x = self.fc(x)

        return x

    def init_weights(self) -> None:
        """
        Initializes the weights of the Transformer model using Xavier initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def to(self, device: torch.device) -> None:
        """
        Moves the model to the specified device.
        :param device: The device to move the model to.
        :type device: torch.device
        """
        super().to(device)

    
    @property
    def model_type(self) -> str:
        """
        Returns the type of model.
        :returns: The model type as a string.
        :rtype: str
        """
        return "neural_network"
        
class PositionalEncoding(nn.Module):
    """
    A class representing the positional encoding used in the Transformer model.
    The positional encoding is added to the input embeddings to inject positional information
    into the sequence.
    :param d_model: The number of expected features in the input (i.e. the embedding dimension).
    :type d_model: int
    :param dropout: The dropout percentage applied to the positional encoding for regularization,
                    preventing overfitting.
    :type dropout: float
    :param max_len: The maximum length of a sequence to be encoded by the positional encoding.
    :type max_len: int
    :example:
        >>> from transformer import PositionalEncoding
        >>> pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=10000)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds the positional encoding to the input tensor.
        :param x: The input tensor.
        :type x: torch.Tensor
        :returns: The output tensor, with the positional encoding added to the input embeddings.
        :rtype: torch.Tensor
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
