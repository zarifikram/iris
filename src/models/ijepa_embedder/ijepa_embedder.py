from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import LossWithIntermediateLosses
from .ijepa import IJEPA
from dataset import Batch


class IJEPA_Embedder(nn.Module):
    def __init__(
        self,
        encoder: IJEPA,
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
        self.patch_size = encoder.patch_size
        self.num_tokens = (encoder.img_size // encoder.patch_size) ** 2
        self.m_start_end = m_start_end

        # define loss
        self.criterion = nn.MSELoss()

    def __repr__(self):
        return "frame_embedder(Assran et al. 2023)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.preprocess_input(x))

    """Update momentum for teacher encoder"""

    def update_momentum(self, m: float):
        student_model = self.model.student_encoder.eval()
        teacher_model = self.model.teacher_encoder.eval()
        with torch.no_grad():
            for student_param, teacher_param in zip(
                student_model.parameters(), teacher_model.parameters()
            ):
                teacher_param.data.mul_(other=m).add_(
                    other=student_param.data, alpha=1 - m
                )

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        observations = self.preprocess_input(
            rearrange(batch["observations"], "b t c h w -> (b t) c h w")
        )
        target_aspect_ratio = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self.model.compute_prediction_and_target(
            observations,
            target_aspect_ratio,
            target_scale,
            context_aspect_ratio,
            context_scale,
        )
        prediction_loss = self.criterion(y_student, y_teacher)
        return LossWithIntermediateLosses(prediction_loss=prediction_loss)

    def teacher_step(self, total_steps: int):
        self.update_momentum(self.m)
        self.m += (self.m_start_end[1] - self.m_start_end[0]) / total_steps

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)
