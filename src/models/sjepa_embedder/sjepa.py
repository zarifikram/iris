import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Set

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, Decoder
from .utils import PatchEmbed, Predictor


class SJEPA(nn.Module):
    def __init__(
        self,
        action_vocab_size: int,
        sequence_length: int,
        in_embed_dim: int,
        embed_dim: int,
        enc_heads: int,
        enc_depth: int,
        decoder_depth: int,
        M: int,
        layer_dropout: float,
        post_emb_norm: bool,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.in_embed_dim = in_embed_dim
        self.embed_dim = embed_dim
        self.M = M
        self.layer_dropout = layer_dropout
        # define the patch embedding and positional embedding
        self.patch_embed = PatchEmbed(
            sequence_length=sequence_length,
            in_embed_dim=in_embed_dim,
            out_embed_dim=embed_dim,
        )
        self.num_tokens = sequence_length
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        self.action_embedding = nn.Embedding(action_vocab_size, embed_dim)
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
        self.predictor = Predictor(self.num_tokens, embed_dim, enc_heads, decoder_depth)

        
    
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
        target_block = torch.zeros(
            (M, x.shape[0], block_h * block_w, x.shape[2]), device=x.device
        )
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
        assert (
            x.ndim == 3
            and x.size(1) == self.sequence_length
            and x.size(2) == self.in_embed_dim
        ), f"Expected input of shape (batch_size, {self.sequence_length}, {self.in_embed_dim}), got {x.shape}"
        x = self.patch_embed(x) + self.pos_embedding
        x = self.post_emb_norm(x)
        return self.student_encoder(x)

    def compute_context_and_target_sequence_embeddings(
        self, frame_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            frame_embeddings.ndim == 3
            and frame_embeddings.size(1) == self.num_tokens + 1
            and frame_embeddings.size(2) == self.in_embed_dim
        )
        context_suquence_embeddings = self.compute_sequence_embeddings(
            frame_embeddings[:, :-1, :], self.student_encoder
        )
        target_sequence_embeddings = self.compute_sequence_embeddings(
            frame_embeddings[:, 1:, :], self.teacher_encoder
        )
        return context_suquence_embeddings, target_sequence_embeddings

    def compute_sequence_embeddings(
        self, frame_embeddings: torch.Tensor, encoder: Encoder
    ) -> torch.Tensor:
        sequence_embeddings = self.patch_embed(frame_embeddings) + self.pos_embedding
        sequence_embeddings = self.norm(sequence_embeddings)
        return encoder(sequence_embeddings)

    def compute_prediction_and_target(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        context_suquence_embeddings, target_sequence_embeddings = (
            self.compute_context_and_target_sequence_embeddings(x)
        )
        action_sequence_embeddings = (
            self.action_embedding(actions.squeeze()) + self.pos_embedding
        )
        assert (
            context_suquence_embeddings.size(1)
            == target_sequence_embeddings.size(1)
            == action_sequence_embeddings.size(1)
        )

        predicted_sequence_embedding, reward, ends = self.predictor(
            context_suquence_embeddings, action_sequence_embeddings
        )

        assert (
            predicted_sequence_embedding.shape == target_sequence_embeddings.shape
        ), f"Prediction shape {predicted_sequence_embedding.shape} must match target shape {target_sequence_embeddings.shape}"

        return predicted_sequence_embedding, reward, ends, target_sequence_embeddings
