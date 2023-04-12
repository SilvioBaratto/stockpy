import torch
from dataclasses import dataclass

@dataclass
class nn_args:
    # neural network
    input_size: int = 4
    hidden_size: int = 8
    output_size: int = 1
    num_layers: int = 2

@dataclass
class prob_args:
    # probabilistic
    input_size : int = 4
    hidden_size : int = 8
    output_size: int = 1
    rnn_dim: int = 32
    z_dim: int = 32
    emission_dim: int = 32
    transition_dim: int = 32
    variance: float = 0.1

@dataclass
class shared:
    # shared
    dropout: float = 0.2
    pretrained: bool = False
    lr: float = 0.001
    betas: tuple = (0.9, 0.999)
    lrd: float = 0.99996
    clip_norm: float = 10.0
    weight_decay: float = 0.001
    eps: float = 1e-8
    amsgrad: bool = False
    optim_args: float = 0.01
    gamma: float = 0.1
    step_size: float = 50

@dataclass
class training:
    # training
    epochs : int = 10
    batch_size : int = 24
    sequence_length : int = 30
    validation_sequence : int = 30
    use_cuda = torch.cuda.is_available()
    num_workers : int = 4
    validation_cadence : int = 5
    patience : int = 5
    
