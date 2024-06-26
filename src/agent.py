from pathlib import Path

from einops import rearrange
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.ijepa_embedder import IJEPA_Embedder
from models.sjepa_embedder import SJEPA_Embedder
from models.actor_critic import ActorCritic
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import extract_state_dict


class Agent(nn.Module):
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, actor_critic: ActorCritic):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        input_ac = obs if self.actor_critic.use_original_obs else torch.clamp(self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)

        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token

class JEPA_Agent(nn.Module):
    def __init__(self, frame_embedder: IJEPA_Embedder, sequence_embedder: SJEPA_Embedder, world_model: WorldModel, actor_critic: ActorCritic): ## TO-DO: change world model to IJEPA_WorldModel
        super().__init__()
        self.frame_embedder = frame_embedder
        self.sequence_embedder = sequence_embedder
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device
    
    def load(self, path_to_checkpoint: Path, device: torch.device, load_embedder: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_embedder:
            self.embedder.load_state_dict(extract_state_dict(agent_state_dict, 'embedder'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        # obs : tensor of (B, T, C, H, W)
        # turn obs to (B, T, E) using IJEPA_Embedder
        # turn obs to (B, T, E) using SJEPA_Embedder
        # pass the output to actor_critic to get logits_actions
        assert self.actor_critic.use_original_obs == False
        input_ac = rearrange(self.frame_embedder(obs), 'b t p p e -> b t (p p e)')
        assert input_ac.shape == (obs.shape[0], obs.shape[1], self.frame_embedder.patch_size**2 * self.frame_embedder.embed_dim)
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token