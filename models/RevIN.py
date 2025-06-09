import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) for time series.
    Normalizes each instance (sample) independently, with optional learnable affine parameters.
    Useful for time series forecasting to remove instance-specific statistics.
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: Number of features/channels in the input.
        :param eps: Small value for numerical stability.
        :param affine: If True, enables learnable affine transformation after normalization.
        :param subtract_last: If True, subtracts the last value in the sequence instead of the mean.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        Forward pass for normalization or denormalization.
        :param x: Input tensor of shape (batch, seq_len, features).
        :param mode: 'norm' for normalization, 'denorm' for denormalization.
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # Initialize learnable affine parameters for each feature.
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # Compute statistics for normalization.
        # Reduce over all dimensions except batch and feature.
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            # Use the last value in the sequence for each instance.
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            # Use the mean across the sequence.
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        # Compute standard deviation for normalization.
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        # Subtract mean or last value.
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        # Divide by standard deviation.
        x = x / self.stdev
        # Optional affine transformation.
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        # Reverse affine transformation if enabled.
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        # Multiply by standard deviation.
        x = x * self.stdev
        # Add back mean or last value.
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
