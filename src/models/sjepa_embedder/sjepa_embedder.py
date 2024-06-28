from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sjepa import SJEPA
from dataset import Batch
from utils import LossWithIntermediateLosses
from models.ijepa_embedder import IJEPA_Embedder


class SJEPA_Embedder(nn.Module):
    def __init__(
        self,
        encoder: SJEPA,
        lr: float,
        weight_decay: float,
        target_aspect_ratio: Tuple[float, float],
        target_scale: Tuple[float, float],
        context_aspect_ratio: float,
        context_scale: Tuple[float, float],
        m: float,
        m_start_end: Tuple[float, float],
    ) -> None:
        super().__init__()

        # define models
        self.model = encoder
        self.in_embed_dim = encoder.in_embed_dim
        # define hyperparameters
        self.M = encoder.M
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale = target_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale
        self.embed_dim = encoder.embed_dim
        self.num_tokens = encoder.sequence_length
        self.m_start_end = m_start_end

        # define loss
        self.criterion = nn.MSELoss()

    def __repr__(self):
        return "sequence_embedder"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.size(-1) == self.in_embed_dim and x.size(-2) == self.num_tokens
        ), f"Expected input of shape (batch_size, {self.num_tokens}, {self.in_embed_dim}), got {x.shape}"
        return self.model(x)

    """Update momentum for teacher encoder"""

    def update_momentum(self, m):
        student_model = self.model.student_encoder.eval()
        teacher_model = self.model.teacher_encoder.eval()
        with torch.no_grad():
            for student_param, teacher_param in zip(
                student_model.parameters(), teacher_model.parameters()
            ):
                teacher_param.data.mul_(other=m).add_(
                    other=student_param.data, alpha=1 - m
                )

    def teacher_step(self, total_steps: int):
        self.update_momentum(self.m)
        self.m += (self.m_start_end[1] - self.m_start_end[0]) / total_steps

    def compute_loss(
        self, batch: Batch, frame_embedder: IJEPA_Embedder, **kwargs: Any
    ) -> LossWithIntermediateLosses:
        with torch.no_grad():
            frame_embeddings = frame_embedder(batch["observations"])

        frame_embeddings = self.preprocess_frame_embeddings(frame_embeddings)
        actions = rearrange(batch["actions"], "b t -> b t 1")[
            :, :-1, :
        ]  # only the first 20 are relevant
        y_student, predicted_reward, predicted_ends, y_teacher = (
            self.model.compute_prediction_and_target(frame_embeddings, actions)
        )
        target_reward, target_ends = self.calculate_target_reward_and_ends(
            batch["rewards"], batch["ends"], batch["mask_padding"]
        )

        prediction_loss = self.criterion(y_student, y_teacher)
        reward_loss = F.cross_entropy(predicted_reward, target_reward)
        ends_loss = F.cross_entropy(predicted_ends, target_ends)
        return LossWithIntermediateLosses(
            prediction_loss=prediction_loss,
            reward_loss=reward_loss,
            ends_loss=ends_loss,
        )

    def calculate_target_reward_and_ends(
        self,
        rewards: torch.Tensor,
        ends: torch.Tensor,
        mask_padding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        mask_fill = torch.logical_not(mask_padding)
        labels_rewards = (
            (rewards.sign() + 1).masked_fill(mask_fill, -100).long()
        )  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)

        # only the last one is relevant
        labels_rewards = labels_rewards[:, -1]
        labels_ends = labels_ends[:, -1]
        return (
            labels_rewards.reshape(-1),
            labels_ends.reshape(-1),
        )

    def calculate_context_and_target_sequence_embeddings(
        self, frame_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            frame_embeddings.ndim == 3
            and frame_embeddings.size(1) == self.num_tokens
            and frame_embeddings.size(2) == self.embed_dim
        )

    def preprocess_frame_embeddings(
        self, frame_embeddings: torch.Tensor
    ) -> torch.Tensor:
        assert frame_embeddings.ndim == 4
        frame_embeddings = rearrange(frame_embeddings, "b t l e -> b t (l e)")
        return frame_embeddings
