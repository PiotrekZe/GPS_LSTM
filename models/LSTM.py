import torch
import torch.nn as nn
from models.RevIN import RevIN


class ConfigurableLSTMModel(nn.Module):
    """
    LSTM model for sequence-to-sequence prediction with configurable input/output lengths and feature dimensions.
    Includes instance normalization (RevIN) for improved time series modeling.
    """

    def __init__(
        self,
        input_seq_len=100,
        output_seq_len=200,
        input_dim=2,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1,
    ):
        """
        Initialize the LSTM model and its components.

        Args:
            input_seq_len (int): Length of input sequence (default: 100)
            output_seq_len (int): Length of output sequence (default: 200)
            input_dim (int): Input feature dimension (default: 2)
            hidden_dim (int): LSTM hidden dimension (default: 128)
            num_layers (int): Number of LSTM layers (default: 1)
            dropout (float): Dropout rate (default: 0.1)
        """
        super(ConfigurableLSTMModel, self).__init__()
        # Instance normalization for input and output
        self.revin_layer = RevIN(input_dim, affine=True, subtract_last=False)

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Main LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Linear projection from hidden state to output feature dimension
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for sequence-to-sequence prediction.
        Args:
            x: Input tensor of shape (batch, input_seq_len, input_dim)
        Returns:
            Output tensor of shape (batch, input_dim, output_seq_len)
        """
        # Permute to (batch, input_dim, input_seq_len) for normalization
        x = x.permute(0, 2, 1)
        # Normalize input instance-wise
        x = self.revin_layer(x, "norm")
        batch_size = x.size(0)
        # LSTM expects (batch, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)

        outputs = []
        # Start decoder with last encoder output
        current_input = lstm_out[:, -1:, :]

        h_decoder = hidden
        c_decoder = cell

        # Autoregressive decoding for output_seq_len steps
        for _ in range(self.output_seq_len):
            projected = self.output_projection(current_input)
            outputs.append(projected)
            # Feed projected output as next input
            lstm_out_step, (h_decoder, c_decoder) = self.lstm(
                projected, (h_decoder, c_decoder)
            )
            current_input = self.dropout(lstm_out_step)

        # Concatenate outputs along time dimension: (batch, output_seq_len, input_dim)
        x = torch.cat(outputs, dim=1)
        # Denormalize output
        x = self.revin_layer(x, "denorm")
        # Permute to (batch, input_dim, output_seq_len) for consistency
        x = x.permute(0, 2, 1)
        return x
