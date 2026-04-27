import torch
import torch.nn as nn
import torch.nn.functional as F

from stockpy.base import EncoderDecoderForecaster
from stockpy.preprocessing import unpack_data
from stockpy.utils import get_activation_function

__all__ = ['TCNForecaster']


class CausalConv1d(nn.Module):
    """Causal 1-D convolution that maintains temporal ordering.

    Left-pads the input by ``(kernel_size - 1) * dilation`` so that the
    output at time *t* only depends on inputs at times ``<= t``.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)
        # out: (batch, out_channels, seq_len)


class TemporalBlock(nn.Module):
    """Residual block with two causal convolutions.

    Each block contains two causal conv layers separated by activation
    and dropout, followed by a residual skip connection.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, bias)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, bias)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """Stack of temporal blocks with exponentially increasing dilation.

    Parameters
    ----------
    n_features : int
        Number of input channels.
    num_filters : int
        Number of output channels per block.
    kernel_size : int
        Kernel size for causal convolutions.
    num_layers : int
        Number of stacked temporal blocks.
    dropout : float, default=0.0
        Dropout rate.
    bias : bool, default=True
        Use bias in conv layers.
    """

    def __init__(
        self,
        n_features,
        num_filters,
        kernel_size,
        num_layers,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = n_features if i == 0 else num_filters
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    bias=bias,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, n_features, seq_len)
        return self.network(x)
        # out: (batch, num_filters, seq_len)


class TCNModel(nn.Module):
    """TCN encoder + GRU decoder for time-series forecasting.

    The TCN encoder processes the context window with causal dilated
    convolutions and residual connections. The final encoder output is
    flattened and projected to initialise the unidirectional GRU
    decoder, which then autoregressively generates the prediction
    horizon. Teacher forcing is supported during training.

    Parameters
    ----------
    n_features : int
        Number of input/output features.
    pred_len : int
        Number of future time steps to predict.
    num_filters : int, default=32
        Number of filters in each TCN block.
    kernel_size : int, default=3
        Kernel size for causal convolutions.
    num_layers : int, default=1
        Number of stacked TCN blocks.
    hidden_size : int, default=32
        Hidden size of the GRU decoder and output projection MLP.
    dropout : float, default=0.0
        Dropout rate.
    activation : str, default='relu'
        Activation for the output projection MLP.
    bias : bool, default=True
        Use bias in conv and linear layers.
    """

    def __init__(
        self,
        n_features,
        pred_len,
        num_filters=32,
        kernel_size=3,
        num_layers=1,
        hidden_size=32,
        dropout=0.0,
        activation='relu',
        bias=True,
    ):
        super().__init__()
        self.n_features = n_features
        self.pred_len = pred_len
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.encoder = TCNEncoder(
            n_features=n_features,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
            bias=bias,
        )

        # Project the flattened last encoder output to decoder hidden size
        self.state_proj = nn.Linear(num_filters, hidden_size, bias=bias)

        self.decoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bias=bias,
            dropout=0.0,
        )

        act = get_activation_function(activation)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=bias),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_features, bias=bias),
        )

    def _init_decoder_state(self, encoder_out):
        """Project the last encoder timestep to decoder initial hidden state.

        encoder_out: (batch, num_filters, seq_len)
        Returns: (1, batch, hidden_size)
        """
        # Take the last timestep: (batch, num_filters)
        last = encoder_out[:, :, -1]
        h = self.state_proj(last)  # (batch, hidden_size)
        return h.unsqueeze(0)  # (1, batch, hidden_size)

    def forward(self, x, y=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, context_len, n_features)
            Past context window.
        y : torch.Tensor or None, shape (batch, pred_len, n_features)
            Ground-truth future values for teacher forcing. If None,
            the decoder runs autoregressively.

        Returns
        -------
        torch.Tensor, shape (batch, pred_len, n_features)
            Forecasted values.
        """
        # Transpose for Conv1d: (batch, n_features, context_len)
        x_t = x.transpose(1, 2)
        enc_out = self.encoder(x_t)
        # enc_out: (batch, num_filters, context_len)

        dec_state = self._init_decoder_state(enc_out)
        # dec_state: (1, batch, hidden_size)

        if y is not None:
            # Teacher forcing: use ground-truth shifted by one timestep
            first_input = x[:, -1:, :]  # (batch, 1, n_features)
            decoder_input = torch.cat([first_input, y[:, :-1, :]], dim=1)
            # decoder_input: (batch, pred_len, n_features)
            out, _ = self.decoder(decoder_input, dec_state)
            # out: (batch, pred_len, hidden_size)
            return self.output_proj(out)
        else:
            # Autoregressive decoding
            decoder_input = x[:, -1:, :]  # (batch, 1, n_features)
            outputs = []
            for _ in range(self.pred_len):
                out, dec_state = self.decoder(decoder_input, dec_state)
                # out: (batch, 1, hidden_size)
                pred = self.output_proj(out)  # (batch, 1, n_features)
                outputs.append(pred)
                decoder_input = pred  # feed previous prediction as next input
            return torch.cat(outputs, dim=1)  # (batch, pred_len, n_features)


class TCNForecaster(EncoderDecoderForecaster):
    """TCN encoder-decoder forecaster with causal dilated convolutions.

    Parameters
    ----------
    num_filters : int, default=32
        Number of filters per TCN residual block.
    kernel_size : int, default=3
        Kernel size for causal convolutions.
    num_layers : int, default=1
        Number of stacked TCN residual blocks.
    hidden_size : int, default=32
        Hidden size of the GRU decoder and output projection.
    dropout : float, default=0.2
        Dropout rate.
    activation : str, default='relu'
        Activation function.
    bias : bool, default=True
        Use bias terms.
    context_len : int, default=20
        Encoder window length.
    pred_len : int, default=1
        Decoder window length.
    **kwargs
        Passed to EncoderDecoderForecaster.
    """

    def __init__(
        self,
        num_filters=32,
        kernel_size=3,
        num_layers=1,
        hidden_size=32,
        dropout=0.2,
        activation='relu',
        bias=True,
        context_len=20,
        pred_len=1,
        **kwargs,
    ):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.bias = bias

        super().__init__(
            context_len=context_len,
            pred_len=pred_len,
            **kwargs,
        )
        self._modules = ['module']

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, TargetTags, InputTags
        return Tags(
            estimator_type='regressor',
            target_tags=TargetTags(required=True, multi_output=True),
            input_tags=InputTags(two_d_array=True),
        )

    def _get_tags(self):
        return {"requires_y": True}

    def initialize_module(self):
        """Create the TCN encoder-decoder module and loss criterion."""
        self.module_ = TCNModel(
            n_features=self.n_features_in_,
            pred_len=self.pred_len,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            bias=self.bias,
        )
        self.criterion_ = nn.MSELoss()
        return self

    def _set_training(self, training=True):
        """Override to set training mode on the PyTorch module."""
        self.module_.train(training)

    def forward(self, x, y=None):
        """Forward pass through the TCN model.

        Parameters
        ----------
        x : torch.Tensor
            Input context window.
        y : torch.Tensor or None
            Target values for teacher forcing.

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, pred_len, n_features).
        """
        return self.module_(x, y=y)

    def train_step_single(self, batch, **fit_params):
        """Override to pass targets to ``infer`` for teacher forcing."""
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        if not self.prob:
            y_pred = self.infer(Xi, y=yi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)
            loss.backward()
            return {'loss': loss, 'y_pred': y_pred}
        return super().train_step_single(batch, **fit_params)

    def state_dict(self):
        """Return the state dict of the underlying PyTorch module."""
        return self.module_.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load a state dict into the underlying PyTorch module."""
        self.module_.load_state_dict(state_dict, strict=strict)

    @property
    def model_type(self):
        return 'cnn'
