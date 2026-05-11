from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PolicyFeatures(BaseFeaturesExtractor):
    """Conv1d over the window + MLP over scalars, fused into one feature vector."""

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
