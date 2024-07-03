from typing import Optional, Union
from einops import rearrange
import numpy as np
import torch
import gym
from torch.distributions.categorical import Categorical
from models.ijepa_embedder import IJEPA_Embedder
from models.sjepa_embedder import SJEPA_Embedder

class JEPAEnv:
    def __init__(
        self,
        frame_embedder: IJEPA_Embedder,
        sequence_embedder: SJEPA_Embedder,
        device: Union[str, torch.device],
        env: Optional[gym.Env] = None,
    ) -> None:
        self.device = torch.device(device)
        self.frame_embedder = frame_embedder.to(self.device).eval()
        self.sequence_embedder = sequence_embedder.to(self.device).eval()

        self.env = env

    @torch.no_grad()
    def reset_from_initial_observations(
        self, initial_observations: torch.Tensor, actions: torch.Tensor = None
    ) -> torch.Tensor:
        # assume observations -> (b, t, c, h, w) -> (b t e)
        assert (
            initial_observations.ndim == 5
            and initial_observations.size(1) == self.sequence_embedder.num_tokens
        )
        frame_embedding = rearrange(
            self.frame_embedder(initial_observations), "b t l e -> b t (l e)"
        )

        self.sequence_embedding = self.sequence_embedder(frame_embedding)
        self.actions = actions if actions is not None else torch.zeros(*self.sequence_embedding.shape[:2], dtype=torch.long, device=self.device)
        return self.sequence_embedding

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor]) -> None:
        self.actions = torch.cat([self.actions[:, 1:], action], dim=1) 
        self.sequence_embedding, reward_logits, ends_logits = (
            self.sequence_embedder.predict_next_state(
                self.sequence_embedding, self.actions
            )
        )

        reward = Categorical(logits=reward_logits).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
        done = Categorical(logits=ends_logits).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)
        return self.sequence_embedding, reward, done, None