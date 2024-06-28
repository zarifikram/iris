from einops import rearrange
import torch
import torch.nn as nn
from x_transformers import Decoder

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if isinstance(img_size, int):
            img_size = img_size, img_size
        if isinstance(patch_size, int):
            patch_size = patch_size, patch_size
        # calculate the number of patches
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        print(self.patch_shape)
        # convolutional layer to convert the image into patches
        self.conv = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.conv(x)
        # flatten the patches
        x = rearrange(x, "b e h w -> b (h w) e")
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