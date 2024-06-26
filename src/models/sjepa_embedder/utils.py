from einops import rearrange
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
        assert x.ndim == 3, "Input must be of shape (batch, sequence, embed_dim)"
        assert x.size(1) == self.sequence_length, "Input sequence length must match the number of blocks"
        x = self.net(x)
        
        return x


"""Lightweight Predictor Module using VIT to predict target patches from context patches"""


class Predictor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()

        self.predictor = Decoder(dim=embed_dim, depth=depth, heads=num_heads)

    def forward(self, context_encoding, target_masks):
        x = torch.cat((context_encoding, target_masks), dim=1)
        x = self.predictor(x)
        # return last len(target_masks) tokens
        l = x.shape[1]
        return x[:, l - target_masks.shape[1] :, :]