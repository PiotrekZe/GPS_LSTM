import torch
import torch.nn as nn
from models.RevIN import RevIN


class AttentionLSTM(nn.Module):
    """
    Temporal LSTM Network with Residual Connections, Bidirectional LSTM Layers, and Attention.
    Includes input normalization (RevIN) and multi-scale prediction fusion for time series forecasting.
    """

    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        num_layers=3,
        seq_len=100,
        pred_len=200,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Ensure hidden_dim is even for bidirectional LSTM splitting
        if hidden_dim % 2 != 0:
            hidden_dim += 1

        # Input projection: 1D convolution + normalization + activation
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.LayerNorm([hidden_dim, seq_len]),
            nn.GELU(),
        )

        # Stacked LSTM layers with alternating bidirectionality and residual connections
        self.lstm_layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        for i in range(num_layers):
            bidirectional = (
                i % 2 == 0
            )  # Alternate between bidirectional and unidirectional LSTM
            self.lstm_layers.append(
                nn.LSTM(
                    hidden_dim,
                    hidden_dim // 2,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            )
            # Projection layer to unify output dimension for residual addition
            proj_in = hidden_dim if bidirectional else hidden_dim // 2
            self.proj_layers.append(nn.Linear(proj_in, hidden_dim))

        # Temporal attention mechanism for sequence modeling
        self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Output network: 1x1 convolutions, activation, and upsampling for prediction
        self.output_net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, output_dim, 1),
            nn.Upsample(size=pred_len, mode="linear", align_corners=True),
        )
        # Instance normalization for input and output
        self.revin_layer = RevIN(input_dim, affine=True, subtract_last=False)

    def forward(self, x, attention=False):
        """
        Forward pass for the TemporalLSTMNetwork.
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            attention: If True, also return attention output and final hidden state
        Returns:
            out: Model predictions (batch, pred_len, output_dim)
            (optionally) attn_out: Attention output
            (optionally) h: Final hidden state after attention
        """
        # Permute to (batch, input_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        # Normalize input instance-wise
        x = self.revin_layer(x, "norm")
        # Permute back to (batch, seq_len, input_dim) for input projection
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        # Permute to (batch, seq_len, hidden_dim) for LSTM
        x = x.permute(0, 2, 1)

        # Pass through stacked LSTM layers with residual connections
        h = x
        for lstm, proj in zip(self.lstm_layers, self.proj_layers):
            out, _ = lstm(h)
            out = proj(out)
            h = out + h  # Residual connection

        # Apply temporal attention and add residual
        attn_out, _ = self.attention(h, h, h)
        h = self.attn_norm(h + attn_out)

        # Prepare for output: (batch, hidden_dim, seq_len)
        h = h.permute(0, 2, 1)
        out = self.output_net(h)
        # Permute to (batch, pred_len, output_dim)
        out = out.permute(0, 2, 1)
        # Denormalize output
        out = self.revin_layer(out, "denorm")
        # Permute to (batch, output_dim, pred_len) for consistency
        out = out.permute(0, 2, 1)
        if attention:
            return out, attn_out, h
        else:
            return out
