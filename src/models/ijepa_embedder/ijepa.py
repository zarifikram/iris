import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Set

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, Decoder
from .utils import PatchEmbed, Predictor


class IJEPA(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        enc_heads: int,
        enc_depth: int,
        decoder_depth: int,
        M: int,
        layer_dropout: float,
        post_emb_norm: bool,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.M = M
        self.layer_dropout = layer_dropout
        # define the patch embedding and positional embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_dim = (
            self.patch_embed.patch_shape[0],
            self.patch_embed.patch_shape[1],
        )
        self.num_tokens = (
            self.patch_embed.patch_shape[0] * self.patch_embed.patch_shape[1]
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        # define the cls and mask tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        # define the encoder and decoder, as well as the layer normalization and dropout
        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.teacher_encoder = Encoder(
            dim=embed_dim,
            heads=enc_heads,
            depth=enc_depth,
            layer_dropout=self.layer_dropout,
        )
        self.student_encoder = copy.deepcopy(self.teacher_encoder)
        self.predictor = Predictor(embed_dim, enc_heads, decoder_depth)

    @torch.no_grad()
    def get_target_block(
        self,
        target_encoder: Encoder,
        x: torch.Tensor,
        patch_dim: Tuple[int, int],
        aspect_ratio: float,
        scale: float,
    ) -> Tuple[torch.Tensor, List[int], Set[int]]:
        
        # get the target block
        target_encoder = target_encoder.eval()
        x = target_encoder(x)
        x = self.norm(x)
        # get the patch dimensions
        patch_h, patch_w = patch_dim
        # get the number of patches
        num_patches = patch_h * patch_w
        # get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        # get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        # get the patches in the target block
        M = self.M
        target_block = torch.zeros((M, x.shape[0], block_h * block_w, x.shape[2]), device=x.device)
        target_patches = []
        for z in range(M):
            # get the starting patch
            start_patch_h = torch.randint(0, patch_h - block_h + 1, (1,)).item()
            start_patch_w = torch.randint(0, patch_w - block_w + 1, (1,)).item()
            start_patch = start_patch_h * patch_w + start_patch_w

            # get the patches in the target block
            patches = (
                torch.arange(block_h).repeat_interleave(block_w) * patch_w
                + torch.arange(block_w).repeat(block_h)
                + start_patch
            )

            # get the target block
            target_patches.append(patches)
            target_block[z] = x[:, patches, :]
        all_patches = torch.cat(target_patches, dim=0).unique().tolist()

        return target_block, target_patches, all_patches

    def get_context_block(
        self,
        x: torch.Tensor,
        patch_dim: Tuple[int, int],
        aspect_ratio: float,
        scale: float,
        target_patches: List[int],
    ) -> torch.Tensor:
        
        patch_h, patch_w = patch_dim
        # get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        # get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        # get the starting patch
        start_patch_h = torch.randint(0, patch_h - block_h + 1, (1,)).item()
        start_patch_w = torch.randint(0, patch_w - block_w + 1, (1,)).item()
        start_patch = start_patch_h * patch_w + start_patch_w
        # get the patches in the context_block
        patches = (
            torch.arange(block_h).repeat_interleave(block_w) * patch_w
            + torch.arange(block_w).repeat(block_h)
            + start_patch
        ).tolist()

        patches = [patch for patch in patches if patch not in target_patches]

        return x[:, patches, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embedding
        x = self.post_emb_norm(x)
        return self.student_encoder(x)

    def compute_prediction_and_target(
        self,
        x: torch.Tensor,
        target_aspect_ratio: float,
        target_scale: float,
        context_aspect_ratio: int,
        context_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get the patch embeddings
        x = self.patch_embed(x)
        b, n, e = x.shape
        # add the positional embeddings
        x = x + self.pos_embedding
        # normalize the embeddings
        x = self.post_emb_norm(x)
        # #get target embeddings
        target_blocks, target_patches, all_patches = self.get_target_block(
            self.teacher_encoder,
            x,
            self.patch_dim,
            target_aspect_ratio,
            target_scale,
        )
        m, b, n, e = target_blocks.shape
        # get context embedding

        context_block = self.get_context_block(
            x, self.patch_dim, context_aspect_ratio, context_scale, all_patches
        )
        context_encoding = self.student_encoder(context_block)
        context_encoding = self.norm(context_encoding)

        prediction_blocks = torch.zeros((m, b, n, e)).to(x.device)
        # get the prediction blocks, predict each target block separately
        for i in range(m):
            target_masks = self.mask_token.repeat(b, n, 1)
            target_pos_embedding = self.pos_embedding[:, target_patches[i], :]
            target_masks = target_masks + target_pos_embedding
            prediction_blocks[i] = self.predictor(context_encoding, target_masks)

        return prediction_blocks, target_blocks
