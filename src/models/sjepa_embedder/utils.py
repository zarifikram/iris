from einops import rearrange
import torch
import torch.nn as nn
from x_transformers import Decoder


class PatchEmbed(nn.Module):
    """Sequence to Embedding"""

    def __init__(self, sequence_length: int, in_embed_dim: int, out_embed_dim: int):
        super().__init__()
        self.sequence_length = sequence_length
        # convolutional layer to convert the image into patches
        self.net = nn.Sequential(
            nn.Linear(in_embed_dim, 2 * out_embed_dim),
            nn.ReLU(),
            nn.Linear(2 * out_embed_dim, out_embed_dim),
        )

    def forward(self, x):
        assert (
            x.size(1) == self.sequence_length
        ), "Input sequence length must match the number of blocks"
        x = self.net(x)

        return x


"""Lightweight Predictor Module using VIT to predict target patches from context patches"""


class Predictor(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int, num_heads: int, depth: int):
        super().__init__()

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.predictor = Decoder(dim=embed_dim, depth=depth, heads=num_heads)
        
        self.reward_head = nn.Sequential(
            nn.Linear(self.num_tokens * 2 * self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 3),
        )

        self.ends_head = nn.Sequential(
            nn.Linear(self.num_tokens * 2 * self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 2),
        )

    def forward(self, context_encoding, action_encoding):
        x = torch.cat((context_encoding, action_encoding), dim=1)
        x = self.predictor(x)
        # return last len(target_masks) tokens
        l = x.shape[1]
        next_sequence_embedding = x[:, l - context_encoding.shape[1] :, :]
        reward = self.reward_head(x.flatten(1))
        ends = self.ends_head(x.flatten(1))
        return next_sequence_embedding, reward, ends
