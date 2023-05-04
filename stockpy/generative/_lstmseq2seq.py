import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from ..config import Config as cfg

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=cfg.nn.input_size, 
                            hidden_size=cfg.nn.hidden_size, 
                            num_layers=cfg.nn.num_layers, 
                            batch_first=True
                            )
        
    def forward(self, x):
        # Forward propagate LSTM
        out, hidden = self.lstm(x)
        
        return out, hidden

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=cfg.nn.input_size,
                            hidden_size=cfg.nn.hidden_size, 
                            num_layers=cfg.nn.num_layers,
                            )
        
        # Fully connected layer
        self.fc = nn.Linear(cfg.nn.hidden_size, 
                            cfg.nn.input_size
                            )
        
    def forward(self, x, hidden):
        # Forward propagate LSTM
        out, hidden = self.lstm(x, hidden)
        
        # Pass LSTM output through fully connected layer
        out = self.fc(out)
        
        return out, hidden

class LSTMSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define encoder and decoder
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, input : torch.Tensor):
        # Pass input tensor through encoder
        encoder_outputs, hidden = self.encoder(input)
        
        # Initialize input tensor for decoder
        input_t = torch.zeros(input.size(0), 
                              cfg.nn.input_size, 
                              dtype=torch.float
                              ).unsqueeze(0)
        
        # Initialize output tensor and hidden state tensor for decoder
        output_tensor = torch.zeros(cfg.training.sequence_length, 
                                    input.size(0), 
                                    cfg.nn.input_size
                                    )
        hidden_states = torch.zeros(cfg.training.sequence_length, 
                                    cfg.nn.num_layers, 
                                    input.size(0), 
                                    cfg.nn.hidden_size
                                    )
        
        # Pass input tensor through decoder at each time step
        for t in range(cfg.training.sequence_length):
            output_t, hidden = self.decoder(input_t, hidden)
            output_t = output_t[-1]
            input_t = output_t.unsqueeze(0)
            output_tensor[t] = output_t
            hidden_states[t] = hidden[0]
        
        return output_tensor, hidden_states

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
        return os.path.basename(os.path.dirname(__file__))