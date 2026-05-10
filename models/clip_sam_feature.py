from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalClipSamFeatureBuilder(nn.Module):
    """
    Build one shared full-class CLIP-SAM feature memory for the whole batch.

    Input shapes:
        clip_image_feat_map_native: [B, D_clip, Hc, Wc]
        clip_text_tokens_native:    [C, K, D_clip]
        sam3_text_tokens_full:      [M, C, D_sam]
        sam3_text_mask_full:        [C, M]

    Output shape:
        shared_clip_feature: [B, N_clip, clip_feature_dim]

    Symbol meanings:
        B means batch size.
        C means full class count.
        K means CLIP prompt template count per class.
        M means SAM3 text token count per class.
        D_clip means native CLIP feature dimension.
        D_sam means SAM3 hidden dimension.
        Hc and Wc mean CLIP image patch grid height and width.
        N_clip means Hc * Wc.
    """

    def __init__(
        self,
        clip_dim: int,
        sam_dim: int,
        clip_feature_dim: int = 256,
        attn_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual_init: float = 0.1,
    ) -> None:
        super().__init__()

        self.clip_dim = int(clip_dim)
        self.sam_dim = int(sam_dim)
        self.clip_feature_dim = int(clip_feature_dim)
        self.attn_dim = int(attn_dim) if attn_dim is not None else int(clip_dim)
        self.num_heads = int(num_heads)

        if self.clip_dim <= 0:
            raise ValueError(f"clip_dim must be positive, got {clip_dim}.")
        if self.sam_dim <= 0:
            raise ValueError(f"sam_dim must be positive, got {sam_dim}.")
        if self.clip_feature_dim <= 0:
            raise ValueError(
                f"clip_feature_dim must be positive, got {clip_feature_dim}."
            )
        if self.attn_dim <= 0:
            raise ValueError(f"attn_dim must be positive, got {attn_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.attn_dim % self.num_heads != 0:
            raise ValueError(
                "attn_dim must be divisible by num_heads, "
                f"got attn_dim={self.attn_dim}, num_heads={self.num_heads}."
            )
        if self.clip_feature_dim % self.num_heads != 0:
            raise ValueError(
                "clip_feature_dim must be divisible by num_heads, "
                f"got clip_feature_dim={self.clip_feature_dim}, "
                f"num_heads={self.num_heads}."
            )

        self.qk_head_dim = self.attn_dim // self.num_heads
        self.v_head_dim = self.clip_feature_dim // self.num_heads

        self.q_proj_clip = nn.Linear(self.clip_dim, self.attn_dim)
        self.k_proj_clip = nn.Linear(self.clip_dim, self.attn_dim)
        self.v_proj_sam = nn.Linear(self.sam_dim, self.clip_feature_dim)

        self.clip_image_to_sam_proj = nn.Linear(
            self.clip_dim,
            self.clip_feature_dim,
        )

        self.dropout = nn.Dropout(float(dropout))
        self.out_norm = nn.LayerNorm(self.clip_feature_dim)

        self.alpha = nn.Parameter(
            torch.tensor(float(residual_init), dtype=torch.float32)
        )

    @staticmethod
    def _masked_mean_sam_tokens(
        sam3_text_tokens_full: torch.Tensor,
        sam3_text_mask_full: torch.Tensor,
    ) -> torch.Tensor:
        if sam3_text_tokens_full.dim() != 3:
            raise ValueError(
                "sam3_text_tokens_full must be [M, C, D_sam], "
                f"got {tuple(sam3_text_tokens_full.shape)}."
            )
        if sam3_text_mask_full.dim() != 2:
            raise ValueError(
                "sam3_text_mask_full must be [C, M], "
                f"got {tuple(sam3_text_mask_full.shape)}."
            )

        seq_len, num_classes, _ = sam3_text_tokens_full.shape
        if tuple(sam3_text_mask_full.shape) != (num_classes, seq_len):
            raise ValueError(
                "SAM3 text mask shape mismatch: expected "
                f"[C, M] = [{num_classes}, {seq_len}], "
                f"got {tuple(sam3_text_mask_full.shape)}."
            )

        tokens = sam3_text_tokens_full.permute(1, 0, 2).contiguous()
        valid = (~sam3_text_mask_full).to(dtype=tokens.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (tokens * valid).sum(dim=1) / denom

    def forward(
        self,
        clip_image_feat_map_native: torch.Tensor,
        clip_text_tokens_native: torch.Tensor,
        sam3_text_tokens_full: torch.Tensor,
        sam3_text_mask_full: torch.Tensor,
    ) -> torch.Tensor:
        if clip_image_feat_map_native.dim() != 4:
            raise ValueError(
                "clip_image_feat_map_native must be [B, D_clip, Hc, Wc], "
                f"got {tuple(clip_image_feat_map_native.shape)}."
            )
        if clip_text_tokens_native.dim() != 3:
            raise ValueError(
                "clip_text_tokens_native must be [C, K, D_clip], "
                f"got {tuple(clip_text_tokens_native.shape)}."
            )

        batch_size, image_dim, grid_h, grid_w = clip_image_feat_map_native.shape
        num_classes, _, text_dim = clip_text_tokens_native.shape

        if int(image_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP image dim mismatch: expected {self.clip_dim}, "
                f"got {image_dim}."
            )
        if int(text_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP text dim mismatch: expected {self.clip_dim}, "
                f"got {text_dim}."
            )
        if int(sam3_text_tokens_full.shape[1]) != int(num_classes):
            raise ValueError(
                "Class count mismatch between CLIP text and SAM3 text: "
                f"{num_classes} vs {sam3_text_tokens_full.shape[1]}."
            )
        if int(sam3_text_tokens_full.shape[2]) != self.sam_dim:
            raise ValueError(
                f"SAM3 text dim mismatch: expected {self.sam_dim}, "
                f"got {sam3_text_tokens_full.shape[2]}."
            )

        device = clip_image_feat_map_native.device
        dtype = clip_image_feat_map_native.dtype

        clip_text_tokens_native = clip_text_tokens_native.to(
            device=device,
            dtype=dtype,
        )
        sam3_text_tokens_full = sam3_text_tokens_full.to(
            device=device,
            dtype=dtype,
        )
        sam3_text_mask_full = sam3_text_mask_full.to(device=device)

        image_tokens = clip_image_feat_map_native.flatten(2).transpose(1, 2)
        image_tokens = image_tokens.contiguous()

        avg_clip_text = clip_text_tokens_native.mean(dim=1)
        avg_sam3_text = self._masked_mean_sam_tokens(
            sam3_text_tokens_full=sam3_text_tokens_full,
            sam3_text_mask_full=sam3_text_mask_full,
        )

        q = self.q_proj_clip(image_tokens)
        k = self.k_proj_clip(avg_clip_text)
        v = self.v_proj_sam(avg_sam3_text)

        num_clip_tokens = int(image_tokens.shape[1])

        # q: [B, N_clip, attn_dim]
        # -> [B, num_heads, N_clip, qk_head_dim]
        q = q.reshape(
            batch_size,
            num_clip_tokens,
            self.num_heads,
            self.qk_head_dim,
        )
        q = q.permute(0, 2, 1, 3).contiguous()

        # k: [C, attn_dim]
        # -> [num_heads, C, qk_head_dim]
        k = k.reshape(
            num_classes,
            self.num_heads,
            self.qk_head_dim,
        )
        k = k.permute(1, 0, 2).contiguous()

        # v: [C, clip_feature_dim]
        # -> [num_heads, C, v_head_dim]
        v = v.reshape(
            num_classes,
            self.num_heads,
            self.v_head_dim,
        )
        v = v.permute(1, 0, 2).contiguous()

        # [B, H, N_clip, D_qk] x [H, C, D_qk]
        # -> [B, H, N_clip, C]
        attn_logits = torch.einsum("bhnd,hcd->bhnc", q, k)
        attn_logits = attn_logits / math.sqrt(float(self.qk_head_dim))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # [B, H, N_clip, C] x [H, C, D_v]
        # -> [B, H, N_clip, D_v]
        attention_out = torch.einsum("bhnc,hcd->bhnd", attn, v)

        # [B, H, N_clip, D_v]
        # -> [B, N_clip, H, D_v]
        # -> [B, N_clip, clip_feature_dim]
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
        attention_out = attention_out.reshape(
            batch_size,
            num_clip_tokens,
            self.clip_feature_dim,
        )

        image_residual = self.clip_image_to_sam_proj(image_tokens)

        shared_clip_feature = image_residual + self.alpha.to(dtype=dtype) * attention_out
        shared_clip_feature = self.out_norm(shared_clip_feature)

        expected_tokens = int(grid_h) * int(grid_w)
        if tuple(shared_clip_feature.shape) != (
            int(batch_size),
            expected_tokens,
            self.clip_feature_dim,
        ):
            raise RuntimeError(
                "shared_clip_feature shape mismatch: expected "
                f"[{batch_size}, {expected_tokens}, {self.clip_feature_dim}], "
                f"got {tuple(shared_clip_feature.shape)}."
            )

        return shared_clip_feature.contiguous()