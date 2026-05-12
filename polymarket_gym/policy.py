from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import jax.numpy as jnp
import flax.linen as fnn


class PolicyFeatures(BaseFeaturesExtractor):
    """Conv1d over the window + MLP over scalars, fused into one feature vector (PyTorch)."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        cnn_channels: int = 32,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        window_space = observation_space.spaces["window"]
        scalar_space = observation_space.spaces["scalars"]
        lookback, n_window_features = window_space.shape
        n_scalars = scalar_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv1d(n_window_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        cnn_out = cnn_channels * 2  # mean-pool + max-pool concatenated

        self.fuse = nn.Sequential(
            nn.Linear(cnn_out + n_scalars, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        window = observations["window"]
        scalars = observations["scalars"]
        # SB3 passes (B, lookback, n_features); Conv1d wants (B, channels, length).
        x = window.transpose(1, 2)
        x = self.cnn(x)
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        return self.fuse(torch.cat([x, scalars], dim=1))


class FlaxPolicyFeatures(fnn.Module):
    """Conv1d over the window + MLP over scalars, fused into one feature vector (Flax/SBX)."""
    features_dim: int
    activation_fn: callable
    lookback: int
    n_window_features: int
    n_scalars: int
    cnn_channels: int = 32

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Gymnasium FlattenObservation sorts dict keys alphabetically.
        # "scalars" comes before "window".
        scalars = x[:, :self.n_scalars]
        window = x[:, self.n_scalars:].reshape(-1, self.lookback, self.n_window_features)

        # Flax Conv default is (batch, steps, features) i.e. channels_last
        c = fnn.Conv(features=self.cnn_channels, kernel_size=(3,), padding=1)(window)
        c = fnn.relu(c)
        c = fnn.Conv(features=self.cnn_channels, kernel_size=(3,), padding=1)(c)
        c = fnn.relu(c)
        
        c_mean = jnp.mean(c, axis=1)
        c_max = jnp.max(c, axis=1)
        
        fuse_in = jnp.concatenate([c_mean, c_max, scalars], axis=1)
        
        # Dense
        f = fnn.Dense(self.features_dim)(fuse_in)
        f = fnn.relu(f)
        f = fnn.Dense(self.features_dim)(f)
        f = fnn.relu(f)
        
        return f
