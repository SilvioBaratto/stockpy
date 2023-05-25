import torch
from dataclasses import dataclass
from typing import List, Union

# Basic settings for all neural networks
@dataclass
class Common:
    hidden_size: Union[int, List[int]] = 32
    num_filters: int = 32
    pool_size: int = 2
    kernel_size: int = 3
    dropout: float = 0.1

# Settings specific to standard neural networks
@dataclass
class NN(Common):
    num_layers: int = 2
    nhead: int = 2

# Settings specific to probabilistic neural networks
@dataclass
class Prob(Common):
    rnn_dim: int = 32
    z_dim: int = 32
    emission_dim: int = 32
    transition_dim: int = 32
    variance: float = 0.1

# Training settings
@dataclass
class Training:
    # Print settings
    eval: bool = False
    
    # Optimizer parameters
    lr: float = 0.001
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.001
    eps: float = 1e-8
    amsgrad: bool = False

    # Scheduler parameters
    gamma: float = 0.1
    step_size: float = 50
    scheduler_patience: int = 5
    min_delta: int = 0.001
    scheduler: bool = True
    scheduler_mode: str = 'min'
    scheduler_factor: float = 0.1
    scheduler_threshold: float = 0.0001
    lrd: float = 0.99996
    clip_norm: float = 10.0

    # Training loop parameters
    scaler_type: str = 'zscore'
    epochs: int = 10
    batch_size: int = 24
    sequence_length: int = 30
    num_workers: int = 4
    validation_cadence: int = 5
    optim_args: float = 0.01
    shuffle: bool = False
    val_size: float = 0.2

    # CUDA settings
    use_cuda = torch.cuda.is_available()
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Miscellaneous settings
    early_stopping: bool = True
    metrics: bool = False
    pretrained: bool = False
    folder: str = None

# The main configuration class that contains all other settings
@dataclass
class Config:
    comm: Common = Common()
    nn: NN = NN()
    prob: Prob = Prob()
    training: Training = Training()

