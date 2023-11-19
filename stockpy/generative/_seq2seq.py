# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from stockpy.base import NumericalGenerator
# from stockpy.base import CategoricalGenerator
# from stockpy.utils import to_device
# from stockpy.utils import get_activation_function

# class Encoder(nn.Module):
#     def __init__(self, 
#                  input_dim, 
#                  hidden_dim, 
#                  n_layers, 
#                  dropout):
        
#         super(Encoder, self).__init__()
        
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
        
#         self.lstm = nn.LSTM(input_dim, 
#                             hidden_dim, 
#                             n_layers, 
#                             batch_first=True)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, src):
#         # Initialize hidden and cell states
#         h0 = torch.zeros(self.n_layers, src.size(0), self.hidden_dim).to(src.device)
#         c0 = torch.zeros(self.n_layers, src.size(0), self.hidden_dim).to(src.device)
        
#         output, (hidden, cell) = self.lstm(self.dropout(src), (h0, c0))
#         return output, hidden, cell

# class Decoder(nn.Module):
#     def __init__(self, 
#                  output_dim, 
#                  hidden_dim, 
#                  n_layers, 
#                  dropout):
        
#         super(Decoder, self).__init__()
        
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.dropout = dropout

#         self.lstm = nn.LSTM(output_dim, 
#                             hidden_dim, 
#                             n_layers, 
#                             batch_first=True)
        
#         self.fc_out = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, input, hidden, cell):

#         output, (hidden, cell) = self.lstm(input, (hidden, cell))
#         output = self.dropout(output)
#         output = self.fc_out(output)
        
#         return output, hidden, cell

    
# class Seq2Seq(nn.Module):
#     def __init__(self, 
#                  hidden_size=32,
#                  num_layers=1,
#                  dropout=0.2,
#                  activation='relu',
#                  bias=True,
#                  seq_len=20,
#                  **kwargs):
        
#         super().__init__()
                
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.activation = activation
#         self.bias = bias
#         self.seq_len = seq_len

#     def initialize_module(self):
#         """
#         Initializes the layers of the neural network based on configuration.
#         """
#         # Checks if hidden_sizes is a single integer and, if so, converts it to a list
#         if isinstance(self.hidden_size, int):
#             self.hidden_sizes = [self.hidden_size]
#         else:
#             self.hidden_sizes = self.hidden_size

#         if isinstance(self, CategoricalGenerator):
#             self.output_size = self.n_classes_
#         elif isinstance(self, NumericalGenerator):
#             self.output_size = self.n_outputs_

#         self.input_size = self.n_features_in_

#         self.encoder = Encoder(input_dim=self.input_size,
#                                hidden_dim=self.hidden_size,
#                                n_layers=self.num_layers,
#                                dropout=self.dropout)
        
#         self.decoder = Decoder(output_dim=self.output_size,
#                                hidden_dim=self.hidden_size,
#                                n_layers=self.num_layers,
#                                dropout=self.dropout)
        
#         to_device(self.encoder, self.device)
#         to_device(self.decoder, self.device)

#     @property
#     def model_type(self):
#         return "seq2seq"
    
# class Seq2SeqNumerical(NumericalGenerator, Seq2Seq):
    
#     def __init__(self,
#                  hidden_size=32,
#                  num_layers=1,
#                  dropout=0.2,
#                  activation='relu',
#                  bias=True,
#                  seq_len=20,
#                  **kwargs):
        
#         NumericalGenerator.__init__(self, **kwargs)
#         Seq2Seq.__init__(self,
#                          hidden_size=hidden_size,
#                          num_layers=num_layers,
#                          dropout=dropout,
#                          activation=activation,
#                          bias=bias,
#                          seq_len=seq_len,
#                          **kwargs
#                          )
        
#         self.criterion = nn.MSELoss()
   
#     def forward(self, src):
#         # Pass src through the Encoder
#         encoder_outputs, hidden, cell = self.encoder(src)

#         # Initialize the first input to the Decoder
#         input = torch.zeros(src.size(0), 1, self.decoder.output_dim).to(src.device)
        
#         outputs = []
        
#         # Loop through each step in the sequence
#         for t in range(self.seq_len):
#             # Pass the input, hidden and cell states to the decoder
#             output, hidden, cell = self.decoder(input, hidden, cell)
            
#             # Append the output (removing the sequence dimension)
#             outputs.append(output.squeeze(1))

#             # Prepare the next input (which is the output in this case)
#             input = output

#         # Stack the outputs into [batch_size, seq_len, output_dim]
#         outputs = torch.stack(outputs, dim=1)

#         return outputs

# class Seq2SeqCategorical(CategoricalGenerator, Seq2Seq):
    
#     def __init__(self,
#                  hidden_size,
#                  num_layers,
#                  dropout,
#                  activation,
#                  bias,
#                  **kwargs):
        
#         CategoricalGenerator.__init__(self, **kwargs)
#         Seq2Seq.__init__(self,
#                          hidden_size=hidden_size,
#                          num_layers=num_layers,
#                          dropout=dropout,
#                          activation=activation,
#                          bias=bias,
#                          **kwargs
#                          )
        
#         self.criterion = nn.NLLLoss()

#     def forward(self, x):
#         pass