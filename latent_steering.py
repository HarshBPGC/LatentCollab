"""
Latent steering utilities for moving one agent's hidden states
toward a baseline agent's alignment region.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SteeringFitStats:
    initial_loss: float
    final_loss: float
    target_distance_before: float
    target_distance_after: float


class LatentSteeringAdapter(nn.Module):
    """
    Learns a light-weight latent adapter that predicts where a source
    agent latent should move in order to match a baseline latent.

    The final steered latent is an interpolation between the original
    latent and the adapter prediction, which helps avoid abrupt policy
    breakage when the Q-head was trained on the source latent manifold.
    """

    def __init__(self, latent_dim: int, alpha: float = 0.6):
        super().__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.mapper = nn.Linear(latent_dim, latent_dim)
        nn.init.eye_(self.mapper.weight)
        nn.init.zeros_(self.mapper.bias)
        self.is_fit = False

    def predict_target(self, source_latents: torch.Tensor) -> torch.Tensor:
        return self.mapper(source_latents)

    def forward(self, source_latents: torch.Tensor) -> torch.Tensor:
        predicted = self.predict_target(source_latents)
        return (1.0 - self.alpha) * source_latents + self.alpha * predicted

    def fit(
        self,
        source_latents: torch.Tensor,
        target_latents: torch.Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
        reg_strength: float = 1e-3,
    ) -> SteeringFitStats:
        if source_latents.shape != target_latents.shape:
            raise ValueError("source_latents and target_latents must have the same shape")
        if source_latents.ndim != 2:
            raise ValueError("Latent matrices must have shape [N, D]")

        source_latents = source_latents.detach()
        target_latents = target_latents.detach()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        with torch.no_grad():
            initial_pred = self.forward(source_latents)
            initial_loss = loss_fn(initial_pred, target_latents).item()
            before_dist = (source_latents - target_latents).norm(dim=-1).mean().item()

        identity = torch.eye(self.latent_dim, device=source_latents.device)
        final_loss = initial_loss

        for _ in range(epochs):
            predicted = self.forward(source_latents)
            fit_loss = loss_fn(predicted, target_latents)
            reg_loss = ((self.mapper.weight - identity) ** 2).mean()
            loss = fit_loss + reg_strength * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        with torch.no_grad():
            final_pred = self.forward(source_latents)
            after_dist = (final_pred - target_latents).norm(dim=-1).mean().item()

        self.is_fit = True
        return SteeringFitStats(
            initial_loss=float(initial_loss),
            final_loss=float(final_loss),
            target_distance_before=float(before_dist),
            target_distance_after=float(after_dist),
        )
