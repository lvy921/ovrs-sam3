from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .task_modes import OUTPUT_KEYS
from .shifted_window_attention import ShiftedWindowAttention2D
from .final_mixer_clip_sam import (
    ClipSamFeatureInitializer,
    CrossGuidedClipSamUpsampler,
    ClipCoarseMaskEmbedder,
)

class ClassTokenBuilder(nn.Module):
    """
    Build per-class trainable class tokens for the final mixer.

    This module owns the learnable class-token query weights. Sam3Image may
    call this module inside the chunk loop, but the weights belong to the
    final mixer.

    Input / output shapes:
        sam3_pair_feats:    [B*C_chunk, M, D]
        sam3_pair_mask:     [B*C_chunk, M]
        class_token_seed:   [B*C_chunk, Q, D]
        class_tokens:       [B*C_chunk, Q, D]

    Symbol meanings:
        B means batch size.
        C_chunk means class count in the current chunk.
        M means SAM3 text token count.
        Q means class token count per class.
        D means SAM3 hidden dimension.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_class_tokens: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_class_tokens = int(num_class_tokens)
        self.num_heads = int(num_heads)

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if self.num_class_tokens <= 0:
            raise ValueError(
                "num_class_tokens must be positive, "
                f"got {num_class_tokens}."
            )
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )

        self.query_embed = nn.Parameter(
            torch.zeros(1, self.num_class_tokens, self.hidden_dim)
        )
        nn.init.normal_(self.query_embed, std=0.02)

        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.text_cross_attn_norm = nn.LayerNorm(self.hidden_dim)

        self.encoder_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.encoder_cross_attn_norm = nn.LayerNorm(self.hidden_dim)

    @staticmethod
    def _sanitize_key_padding_mask(
        key_padding_mask: Optional[torch.Tensor],
        expected_shape: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if key_padding_mask is None:
            return None

        if tuple(key_padding_mask.shape) != tuple(expected_shape):
            raise ValueError(
                "key_padding_mask shape mismatch: expected "
                f"{expected_shape}, got {tuple(key_padding_mask.shape)}."
            )

        key_padding_mask = key_padding_mask.detach().bool()

        # MultiheadAttention can produce NaN if one row is fully masked.
        fully_masked = key_padding_mask.all(dim=1)
        if fully_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_masked, 0] = False

        return key_padding_mask.contiguous()

    def build_seed_from_sam3_text(
        self,
        sam3_pair_feats: torch.Tensor,
        sam3_pair_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if sam3_pair_feats.dim() != 3:
            raise ValueError(
                "sam3_pair_feats must be [B*C_chunk, M, D], "
                f"got {tuple(sam3_pair_feats.shape)}."
            )

        pair_count, seq_len, feat_dim = sam3_pair_feats.shape
        if int(feat_dim) != self.hidden_dim:
            raise ValueError(
                f"sam3_pair_feats dim mismatch: expected {self.hidden_dim}, "
                f"got {feat_dim}."
            )

        sam3_pair_mask = self._sanitize_key_padding_mask(
            key_padding_mask=sam3_pair_mask,
            expected_shape=(int(pair_count), int(seq_len)),
        )

        query_embed = self.query_embed.to(
            device=sam3_pair_feats.device,
            dtype=sam3_pair_feats.dtype,
        )
        query_embed = query_embed.expand(
            int(pair_count),
            self.num_class_tokens,
            self.hidden_dim,
        )

        sam3_pair_feats = sam3_pair_feats.detach()

        attn_out, _ = self.text_cross_attn(
            query=query_embed,
            key=sam3_pair_feats,
            value=sam3_pair_feats,
            key_padding_mask=sam3_pair_mask,
            need_weights=False,
        )

        class_token_seed = self.text_cross_attn_norm(query_embed + attn_out)
        return class_token_seed.contiguous()

    @staticmethod
    def _prepare_encoder_tokens(
        encoder_hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        num_pairs: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states.shape[0] == num_pairs:
            encoder_tokens = encoder_hidden_states.contiguous()
        elif encoder_hidden_states.shape[1] == num_pairs:
            encoder_tokens = encoder_hidden_states.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "Cannot infer encoder token layout: "
                f"encoder_hidden_states.shape={tuple(encoder_hidden_states.shape)}, "
                f"num_pairs={num_pairs}."
            )

        if padding_mask is not None:
            expected_shape = (int(num_pairs), int(encoder_tokens.shape[1]))
            if tuple(padding_mask.shape) != expected_shape:
                raise ValueError(
                    "padding_mask shape mismatch: expected "
                    f"{expected_shape}, got {tuple(padding_mask.shape)}."
                )
            padding_mask = padding_mask.detach().bool().contiguous()

            fully_masked = padding_mask.all(dim=1)
            if fully_masked.any():
                padding_mask = padding_mask.clone()
                padding_mask[fully_masked, 0] = False

        return encoder_tokens, padding_mask

    def refine_with_encoder_memory(
        self,
        class_token_seed: torch.Tensor,
        encoder_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if class_token_seed.dim() != 3:
            raise ValueError(
                "class_token_seed must be [B*C_chunk, Q, D], "
                f"got {tuple(class_token_seed.shape)}."
            )

        if int(class_token_seed.shape[-1]) != self.hidden_dim:
            raise ValueError(
                f"class_token_seed dim mismatch: expected {self.hidden_dim}, "
                f"got {class_token_seed.shape[-1]}."
            )

        num_pairs = int(class_token_seed.shape[0])

        encoder_tokens, encoder_padding_mask = self._prepare_encoder_tokens(
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            padding_mask=encoder_out.get("padding_mask", None),
            num_pairs=num_pairs,
        )

        encoder_tokens = encoder_tokens.detach()

        attn_out, _ = self.encoder_cross_attn(
            query=class_token_seed,
            key=encoder_tokens,
            value=encoder_tokens,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
        )

        class_tokens = self.encoder_cross_attn_norm(class_token_seed + attn_out)
        return class_tokens.contiguous()

class MaskEmbeddingFusionLayer(nn.Module):
    """
    One layer of mask-embedding fusion.

    New design:
        1. Update class tokens with slot-wise inter-class self-attention.
        2. Update class tokens with intra-class self-attention.
        3. Fuse CLIP-SAM feature and current mask embedding by window attention.
        4. Let class tokens attend the refined feature.
        5. Predict presence logits.
        6. Build semantic prior embedding with presence-signed scaling.
        7. Add semantic prior embedding to refined mask embedding and normalize.

    Input:
        class_tokens:          [B, C, Q, D]
        semantic_logits:       [B, C, H, W]
        clip_sam_feature_high: [B, H*W, D]
        mask_embed:            [B, D, H, W]
        class_code:            [B, C, D]

    Output:
        class_tokens:          [B, C, Q, D]
        presence_logits:       [B, C]
        mask_embed:            [B, D, H, W]

    Symbol meanings:
        B means batch size.
        C means class count.
        Q means class token count per class.
        D means hidden feature dimension.
        H and W mean mask height and width.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        presence_enabled: bool = True,
        window_size: int = 8,
        shift_size: int = 0,
        class_feature_pool_stride: int = 4,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.presence_enabled = bool(presence_enabled)
        self.class_feature_pool_stride = int(class_feature_pool_stride)

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )
        if self.class_feature_pool_stride <= 0:
            raise ValueError(
                "class_feature_pool_stride must be positive, "
                f"got {class_feature_pool_stride}."
            )

        self.slot_inter_class_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.slot_inter_class_norm = nn.LayerNorm(self.hidden_dim)

        self.intra_class_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.intra_class_norm = nn.LayerNorm(self.hidden_dim)

        self.mask_feature_attn = ShiftedWindowAttention2D(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            window_size=int(window_size),
            shift_size=int(shift_size),
            dropout=float(dropout),
            value_preserving=True,
            residual_source="value",
            use_residual_norm=True,
            use_rel_pos_bias=True,
        )

        self.class_to_feature_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.class_to_feature_norm = nn.LayerNorm(self.hidden_dim)

        self.presence_query = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.presence_query, std=0.02)

        self.presence_token_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.presence_token_norm = nn.LayerNorm(self.hidden_dim)

        self.presence_summary_norm = nn.LayerNorm(self.hidden_dim * 3)
        self.presence_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

        self.mask_embed_update_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _normalize_map(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"x must be [B, D, H, W], got {tuple(x.shape)}."
            )

        batch_size, dim, height, width = x.shape
        x_dtype = x.dtype

        x = x.flatten(2).transpose(1, 2).contiguous()
        x = norm(x)
        x = x.transpose(1, 2).reshape(batch_size, dim, height, width)
        return x.to(dtype=x_dtype).contiguous()

    def _slot_wise_inter_class_self_attn(
        self,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes, num_tokens, dim = class_tokens.shape

        x = class_tokens.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * num_tokens, num_classes, dim)

        delta, _ = self.slot_inter_class_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )
        x = self.slot_inter_class_norm(x + self.dropout(delta))

        x = x.reshape(batch_size, num_tokens, num_classes, dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _intra_class_self_attn(
        self,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes, num_tokens, dim = class_tokens.shape

        x = class_tokens.reshape(batch_size * num_classes, num_tokens, dim)

        delta, _ = self.intra_class_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )
        x = self.intra_class_norm(x + self.dropout(delta))

        return x.reshape(batch_size, num_classes, num_tokens, dim).contiguous()

    def _pool_feature_for_class_attention(
        self,
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        if feature_map.dim() != 4:
            raise ValueError(
                "feature_map must be [B, D, H, W], "
                f"got {tuple(feature_map.shape)}."
            )

        stride = int(self.class_feature_pool_stride)
        if stride <= 1:
            return feature_map

        return F.avg_pool2d(
            feature_map,
            kernel_size=stride,
            stride=stride,
            ceil_mode=True,
            count_include_pad=False,
        )

    def _attend_feature_with_class_tokens(
        self,
        class_tokens: torch.Tensor,
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D], "
                f"got {tuple(class_tokens.shape)}."
            )
        if feature_map.dim() != 4:
            raise ValueError(
                "feature_map must be [B, D, H, W], "
                f"got {tuple(feature_map.shape)}."
            )

        batch_size, num_classes, num_tokens, dim = class_tokens.shape
        feature_batch, feature_dim, _, _ = feature_map.shape

        if int(feature_batch) != int(batch_size):
            raise ValueError(
                f"feature batch mismatch: {feature_batch} vs {batch_size}."
            )
        if int(feature_dim) != int(dim):
            raise ValueError(
                f"feature dim mismatch: {feature_dim} vs {dim}."
            )

        pooled_feature = self._pool_feature_for_class_attention(feature_map)
        feature_tokens = pooled_feature.flatten(2).transpose(1, 2).contiguous()
        num_pixels = int(feature_tokens.shape[1])

        query = class_tokens.reshape(batch_size * num_classes, num_tokens, dim)

        key = feature_tokens[:, None].expand(
            batch_size,
            num_classes,
            num_pixels,
            dim,
        )
        key = key.reshape(batch_size * num_classes, num_pixels, dim)
        value = key

        attn_out, _ = self.class_to_feature_attn(
            query=query,
            key=key,
            value=value,
            need_weights=False,
        )

        out = self.class_to_feature_norm(query + self.dropout(attn_out))
        return out.reshape(batch_size, num_classes, num_tokens, dim).contiguous()

    def _build_presence_logits(
        self,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D], "
                f"got {tuple(class_tokens.shape)}."
            )

        batch_size, num_classes, num_tokens, dim = class_tokens.shape
        if int(dim) != self.hidden_dim:
            raise ValueError(
                f"class token dim mismatch: expected {self.hidden_dim}, "
                f"got {dim}."
            )

        x = class_tokens.reshape(
            batch_size * num_classes,
            num_tokens,
            dim,
        )

        query = self.presence_query.to(
            device=class_tokens.device,
            dtype=class_tokens.dtype,
        )
        query = query.expand(batch_size * num_classes, 1, dim)

        attn_out, _ = self.presence_token_attn(
            query=query,
            key=x,
            value=x,
            need_weights=False,
        )
        attn_summary = self.presence_token_norm(
            query + self.dropout(attn_out)
        ).squeeze(1)

        mean_summary = x.mean(dim=1)
        max_summary = x.max(dim=1).values

        summary = torch.cat(
            [
                attn_summary,
                mean_summary,
                max_summary,
            ],
            dim=-1,
        )
        summary = self.presence_summary_norm(summary)

        presence_logits = self.presence_head(summary).squeeze(-1)
        return presence_logits.reshape(batch_size, num_classes).contiguous()

    def _build_presence_signed_semantic_prior_embedding(
        self,
        semantic_logits: torch.Tensor,
        presence_logits: torch.Tensor,
        class_code: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )
        if presence_logits.dim() != 2:
            raise ValueError(
                "presence_logits must be [B, C], "
                f"got {tuple(presence_logits.shape)}."
            )
        if class_code.dim() != 3:
            raise ValueError(
                "class_code must be [B, C, D], "
                f"got {tuple(class_code.shape)}."
            )

        batch_size, num_classes, _, _ = semantic_logits.shape

        if tuple(presence_logits.shape) != (batch_size, num_classes):
            raise ValueError(
                "presence_logits shape mismatch: expected "
                f"{(batch_size, num_classes)}, got {tuple(presence_logits.shape)}."
            )
        if tuple(class_code.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "class_code batch/class mismatch: expected "
                f"{(batch_size, num_classes)}, got {tuple(class_code.shape[:2])}."
            )

        semantic_logits = semantic_logits.to(
            device=class_code.device,
            dtype=class_code.dtype,
        )
        presence_logits = presence_logits.to(
            device=class_code.device,
            dtype=class_code.dtype,
        )

        if self.presence_enabled:
            presence_score = torch.sigmoid(presence_logits)
        else:
            presence_score = semantic_logits.new_ones(batch_size, num_classes)

        presence_score = presence_score[:, :, None, None]

        positive_scale = presence_score
        negative_scale = 2.0 - presence_score

        signed_scale = torch.where(
            semantic_logits >= 0,
            positive_scale,
            negative_scale,
        )
        adjusted_logits = semantic_logits * signed_scale

        mask_prob = torch.softmax(adjusted_logits, dim=1)

        prior_embed = torch.einsum(
            "bchw,bcd->bdhw",
            mask_prob,
            class_code,
        )

        return prior_embed.contiguous()

    def forward(
        self,
        class_tokens: torch.Tensor,
        semantic_logits: torch.Tensor,
        clip_sam_feature_high: torch.Tensor,
        mask_embed: torch.Tensor,
        class_code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D], "
                f"got {tuple(class_tokens.shape)}."
            )
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )
        if clip_sam_feature_high.dim() != 3:
            raise ValueError(
                "clip_sam_feature_high must be [B, H*W, D], "
                f"got {tuple(clip_sam_feature_high.shape)}."
            )
        if mask_embed.dim() != 4:
            raise ValueError(
                "mask_embed must be [B, D, H, W], "
                f"got {tuple(mask_embed.shape)}."
            )
        if class_code.dim() != 3:
            raise ValueError(
                "class_code must be [B, C, D], "
                f"got {tuple(class_code.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape
        token_batch, token_classes, _, dim = class_tokens.shape

        if (token_batch, token_classes) != (batch_size, num_classes):
            raise ValueError(
                "class_tokens and semantic_logits batch/class mismatch: "
                f"{tuple(class_tokens.shape[:2])} vs {(batch_size, num_classes)}."
            )
        if tuple(class_code.shape) != (batch_size, num_classes, dim):
            raise ValueError(
                "class_code shape mismatch: expected "
                f"{(batch_size, num_classes, dim)}, got {tuple(class_code.shape)}."
            )
        if tuple(mask_embed.shape) != (batch_size, dim, height, width):
            raise ValueError(
                "mask_embed shape mismatch: expected "
                f"{(batch_size, dim, height, width)}, got {tuple(mask_embed.shape)}."
            )
        if tuple(clip_sam_feature_high.shape) != (
            batch_size,
            height * width,
            dim,
        ):
            raise ValueError(
                "clip_sam_feature_high shape mismatch: expected "
                f"{(batch_size, height * width, dim)}, "
                f"got {tuple(clip_sam_feature_high.shape)}."
            )
        if int(dim) != self.hidden_dim:
            raise ValueError(
                f"class token dim mismatch: expected {self.hidden_dim}, got {dim}."
            )

        semantic_logits = semantic_logits.to(
            device=class_tokens.device,
            dtype=class_tokens.dtype,
        )
        clip_sam_feature_high = clip_sam_feature_high.to(
            device=class_tokens.device,
            dtype=class_tokens.dtype,
        )
        mask_embed = mask_embed.to(
            device=class_tokens.device,
            dtype=class_tokens.dtype,
        )
        class_code = class_code.to(
            device=class_tokens.device,
            dtype=class_tokens.dtype,
        )

        class_tokens = self._slot_wise_inter_class_self_attn(class_tokens)
        class_tokens = self._intra_class_self_attn(class_tokens)

        clip_map = clip_sam_feature_high.transpose(1, 2).reshape(
            batch_size,
            dim,
            height,
            width,
        )

        refined_mask_embed = self.mask_feature_attn(
            query_map=clip_map,
            key_map=mask_embed,
            value_map=mask_embed,
        )

        class_tokens = self._attend_feature_with_class_tokens(
            class_tokens=class_tokens,
            feature_map=refined_mask_embed,
        )

        if self.presence_enabled:
            presence_logits = self._build_presence_logits(class_tokens)
        else:
            presence_logits = semantic_logits.new_zeros(batch_size, num_classes)

        prior_embed = self._build_presence_signed_semantic_prior_embedding(
            semantic_logits=semantic_logits,
            presence_logits=presence_logits,
            class_code=class_code,
        )

        updated_mask_embed = self._normalize_map(
            self.mask_embed_update_norm,
            refined_mask_embed + prior_embed,
        )

        return (
            class_tokens.contiguous(),
            presence_logits.contiguous(),
            updated_mask_embed.contiguous(),
        )


class ClassTokenSemanticFinalMixer(nn.Module):
    """
    Final mixer for open-vocabulary semantic segmentation.

    New design:
        1. Own class-token query weights through ClassTokenBuilder.
        2. Build CLIP-SAM feature inside final mixer.
        3. Build class_code by averaging class tokens.
        4. Build one initial SAM3 semantic prior mask embedding.
        5. Update the same mask embedding through multiple fusion layers.
        6. Use presence-signed semantic prior in each layer.
        7. Produce every layer's mask logits by dot(mask_embed, initial class_code).

    Input:
        semantic_logits:              [B, C, H, W]
        class_tokens:                 [B, C, Q, D_sam]
        clip_image_feat_map_native:   [B, D_clip, Hc, Wc]
        clip_text_tokens_native:      [C, K, D_clip]
        sam3_feature_high:            [B, D_sam, H, W]
        clip_grid_hw:                 (Hc, Wc)

    Output:
        final_logits:                 [B, C, H, W]
        mask_logits_layers:           [L, B, C, H, W]
        presence_logits:              [B, C]
        presence_score:               [B, C]
        presence_logits_layers:       [L, B, C]
        clip_coarse_logits:           [B, C, H, W]
        clip_coarse_pred:             [B, H, W]

    Symbol meanings:
        B means batch size.
        C means class count.
        Q means class token count per class.
        K means CLIP prompt-template count per class.
        D_sam means SAM3 hidden dimension.
        D_clip means CLIP feature dimension.
        H and W mean final mask height and width.
        Hc and Wc mean CLIP feature grid height and width.
        L means fusion layer count.
    """

    def __init__(
        self,
        sam_dim: int,
        clip_dim: int,
        num_class_tokens: int = 32,
        num_heads: int = 8,
        fusion_layers: int = 4,
        dropout: float = 0.1,
        presence_enabled: bool = True,
        tau_mask: float = 16.0,
        clip_sam_feature_enabled: bool = True,
        clip_sam_upsample_enabled: bool = True,
        clip_sam_upsample_window_size: int = 8,
        clip_sam_upsample_shift_size: int = 4,
        clip_sam_upsample_dropout: float = 0.1,
        window_size: int = 8,
        shift_size: int = 4,
        window_dropout: float = 0.1,
        class_feature_pool_stride: int = 4,
    ) -> None:
        super().__init__()

        self.sam_dim = int(sam_dim)
        self.clip_dim = int(clip_dim)
        self.num_class_tokens = int(num_class_tokens)
        self.num_heads = int(num_heads)
        self.fusion_layers = int(fusion_layers)
        self.presence_enabled = bool(presence_enabled)
        self.tau_mask = float(tau_mask)

        self.clip_sam_feature_enabled = bool(clip_sam_feature_enabled)
        self.clip_sam_upsample_enabled = bool(clip_sam_upsample_enabled)

        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.window_dropout = float(window_dropout)
        self.class_feature_pool_stride = int(class_feature_pool_stride)

        if self.sam_dim <= 0:
            raise ValueError(f"sam_dim must be positive, got {sam_dim}.")
        if self.clip_dim <= 0:
            raise ValueError(f"clip_dim must be positive, got {clip_dim}.")
        if self.num_class_tokens <= 0:
            raise ValueError(
                "num_class_tokens must be positive, "
                f"got {num_class_tokens}."
            )
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.fusion_layers <= 0:
            raise ValueError(f"fusion_layers must be positive, got {fusion_layers}.")
        if self.sam_dim % self.num_heads != 0:
            raise ValueError(
                "sam_dim must be divisible by num_heads, "
                f"got sam_dim={self.sam_dim}, num_heads={self.num_heads}."
            )
        if self.clip_dim % self.num_heads != 0:
            raise ValueError(
                "clip_dim must be divisible by num_heads, "
                f"got clip_dim={self.clip_dim}, num_heads={self.num_heads}."
            )
        if self.tau_mask <= 0:
            raise ValueError(f"tau_mask must be positive, got {self.tau_mask}.")
        if not self.clip_sam_feature_enabled:
            raise ValueError("clip_sam_feature_enabled=False is not supported.")
        if not self.clip_sam_upsample_enabled:
            raise ValueError("clip_sam_upsample_enabled=False is not supported.")
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                "shift_size must satisfy 0 <= shift_size < window_size, "
                f"got shift_size={self.shift_size}, window_size={self.window_size}."
            )
        if self.class_feature_pool_stride <= 0:
            raise ValueError(
                "class_feature_pool_stride must be positive, "
                f"got {self.class_feature_pool_stride}."
            )

        self.class_token_builder = ClassTokenBuilder(
            hidden_dim=self.sam_dim,
            num_class_tokens=self.num_class_tokens,
            num_heads=self.num_heads,
            dropout=float(dropout),
        )

        self.clip_sam_initializer = ClipSamFeatureInitializer(
            clip_dim=self.clip_dim,
            sam_dim=self.sam_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
        )

        self.clip_sam_upsampler = CrossGuidedClipSamUpsampler(
            hidden_dim=self.sam_dim,
            num_heads=self.num_heads,
            window_size=int(clip_sam_upsample_window_size),
            shift_size=int(clip_sam_upsample_shift_size),
            dropout=float(clip_sam_upsample_dropout),
        )

        self.clip_coarse_embedder = ClipCoarseMaskEmbedder(
            clip_dim=self.clip_dim,
            sam_dim=self.sam_dim,
        )

        self.class_code_norm = nn.LayerNorm(self.sam_dim)
        self.initial_mask_embed_norm = nn.LayerNorm(self.sam_dim)

        layers = []
        for layer_idx in range(self.fusion_layers):
            layer_shift_size = 0 if layer_idx % 2 == 0 else self.shift_size

            layers.append(
                MaskEmbeddingFusionLayer(
                    hidden_dim=self.sam_dim,
                    num_heads=self.num_heads,
                    dropout=self.window_dropout,
                    presence_enabled=self.presence_enabled,
                    window_size=self.window_size,
                    shift_size=layer_shift_size,
                    class_feature_pool_stride=self.class_feature_pool_stride,
                )
            )

        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _normalize_map(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"x must be [B, D, H, W], got {tuple(x.shape)}."
            )

        batch_size, dim, height, width = x.shape
        x_dtype = x.dtype

        x = x.flatten(2).transpose(1, 2).contiguous()
        x = norm(x)
        x = x.transpose(1, 2).reshape(batch_size, dim, height, width)
        return x.to(dtype=x_dtype).contiguous()

    def build_class_token_seed_from_sam3_text(
        self,
        sam3_pair_feats: torch.Tensor,
        sam3_pair_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.class_token_builder.build_seed_from_sam3_text(
            sam3_pair_feats=sam3_pair_feats,
            sam3_pair_mask=sam3_pair_mask,
        )

    def run_class_token_encoder_cross_attn(
        self,
        class_token_seed: torch.Tensor,
        encoder_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.class_token_builder.refine_with_encoder_memory(
            class_token_seed=class_token_seed,
            encoder_out=encoder_out,
        )

    def _build_class_code(
        self,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D], "
                f"got {tuple(class_tokens.shape)}."
            )

        batch_size, num_classes, _, dim = class_tokens.shape
        if int(dim) != self.sam_dim:
            raise ValueError(
                f"class token dim mismatch: expected {self.sam_dim}, got {dim}."
            )

        class_code = class_tokens.mean(dim=2)
        class_code = self.class_code_norm(class_code)
        return class_code.reshape(batch_size, num_classes, dim).contiguous()

    def _build_initial_mask_embedding(
        self,
        semantic_logits: torch.Tensor,
        class_code: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )
        if class_code.dim() != 3:
            raise ValueError(
                "class_code must be [B, C, D], "
                f"got {tuple(class_code.shape)}."
            )

        batch_size, num_classes, _, _ = semantic_logits.shape
        if tuple(class_code.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "class_code batch/class mismatch: expected "
                f"{(batch_size, num_classes)}, got {tuple(class_code.shape[:2])}."
            )

        mask_prob = torch.softmax(semantic_logits, dim=1)

        mask_embed = torch.einsum(
            "bchw,bcd->bdhw",
            mask_prob,
            class_code,
        ).contiguous()

        mask_embed = self._normalize_map(self.initial_mask_embed_norm, mask_embed)
        return mask_embed.contiguous()

    def _build_mask_logits(
        self,
        mask_embed: torch.Tensor,
        class_code: torch.Tensor,
    ) -> torch.Tensor:
        if mask_embed.dim() != 4:
            raise ValueError(
                "mask_embed must be [B, D, H, W], "
                f"got {tuple(mask_embed.shape)}."
            )
        if class_code.dim() != 3:
            raise ValueError(
                "class_code must be [B, C, D], "
                f"got {tuple(class_code.shape)}."
            )

        batch_size, dim, height, width = mask_embed.shape
        code_batch, num_classes, code_dim = class_code.shape

        if int(code_batch) != int(batch_size):
            raise ValueError(
                f"class_code batch mismatch: {code_batch} vs {batch_size}."
            )
        if int(code_dim) != int(dim):
            raise ValueError(
                f"class_code dim mismatch: {code_dim} vs {dim}."
            )

        mask_tokens = mask_embed.flatten(2).transpose(1, 2).contiguous()

        raw_logits = torch.einsum(
            "bnd,bcd->bcn",
            mask_tokens,
            class_code,
        )

        mask_logits = raw_logits / float(self.tau_mask)

        return mask_logits.reshape(
            batch_size,
            num_classes,
            height,
            width,
        ).contiguous()

    def _validate_inputs(
        self,
        semantic_logits: torch.Tensor,
        class_tokens: torch.Tensor,
        clip_image_feat_map_native: torch.Tensor,
        clip_text_tokens_native: torch.Tensor,
        sam3_feature_high: torch.Tensor,
        clip_grid_hw: tuple[int, int],
    ) -> None:
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D_sam], "
                f"got {tuple(class_tokens.shape)}."
            )
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
        if sam3_feature_high.dim() != 4:
            raise ValueError(
                "sam3_feature_high must be [B, D_sam, H, W], "
                f"got {tuple(sam3_feature_high.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape
        token_batch, token_classes, token_count, token_dim = class_tokens.shape
        clip_batch, clip_dim, clip_h, clip_w = clip_image_feat_map_native.shape
        text_classes, _, text_dim = clip_text_tokens_native.shape

        if tuple(class_tokens.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "class_tokens batch/class mismatch: expected "
                f"{(batch_size, num_classes)}, got {tuple(class_tokens.shape[:2])}."
            )
        if int(token_count) != self.num_class_tokens:
            raise ValueError(
                f"class token count mismatch: expected {self.num_class_tokens}, "
                f"got {token_count}."
            )
        if int(token_dim) != self.sam_dim:
            raise ValueError(
                f"class token dim mismatch: expected {self.sam_dim}, "
                f"got {token_dim}."
            )
        if int(clip_batch) != int(batch_size):
            raise ValueError(
                f"CLIP image batch mismatch: {clip_batch} vs {batch_size}."
            )
        if int(clip_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP image dim mismatch: expected {self.clip_dim}, got {clip_dim}."
            )
        if int(text_classes) != int(num_classes):
            raise ValueError(
                f"CLIP text class count mismatch: {text_classes} vs {num_classes}."
            )
        if int(text_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP text dim mismatch: expected {self.clip_dim}, got {text_dim}."
            )
        if tuple(sam3_feature_high.shape) != (
            batch_size,
            self.sam_dim,
            height,
            width,
        ):
            raise ValueError(
                "sam3_feature_high shape mismatch: expected "
                f"{(batch_size, self.sam_dim, height, width)}, "
                f"got {tuple(sam3_feature_high.shape)}."
            )

        expected_clip_grid_hw = (int(clip_h), int(clip_w))
        if tuple(int(x) for x in clip_grid_hw) != expected_clip_grid_hw:
            raise ValueError(
                "clip_grid_hw mismatch: expected "
                f"{expected_clip_grid_hw}, got {clip_grid_hw}."
            )

    def forward(
        self,
        semantic_logits: torch.Tensor,
        class_tokens: torch.Tensor,
        clip_image_feat_map_native: torch.Tensor,
        clip_text_tokens_native: torch.Tensor,
        sam3_feature_high: torch.Tensor,
        clip_grid_hw: tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        self._validate_inputs(
            semantic_logits=semantic_logits,
            class_tokens=class_tokens,
            clip_image_feat_map_native=clip_image_feat_map_native,
            clip_text_tokens_native=clip_text_tokens_native,
            sam3_feature_high=sam3_feature_high,
            clip_grid_hw=clip_grid_hw,
        )

        batch_size, num_classes, height, width = semantic_logits.shape

        device = class_tokens.device
        dtype = class_tokens.dtype

        semantic_logits = semantic_logits.detach().to(device=device, dtype=dtype)
        clip_image_feat_map_native = clip_image_feat_map_native.to(
            device=device,
            dtype=dtype,
        )
        clip_text_tokens_native = clip_text_tokens_native.to(
            device=device,
            dtype=dtype,
        )
        sam3_feature_high = sam3_feature_high.detach().to(
            device=device,
            dtype=dtype,
        )

        # Fixed class code for the whole final mixer.
        # Later class tokens can be updated, but mask logits always use this
        # initial class_code.
        class_code = self._build_class_code(class_tokens)

        aligned_clip_sam_feature_low = self.clip_sam_initializer(
            clip_image_feat_map_native=clip_image_feat_map_native,
            clip_text_tokens_native=clip_text_tokens_native,
            class_token_query_embed=self.class_token_builder.query_embed,
            class_tokens=class_tokens,
        )

        clip_sam_feature_high = self.clip_sam_upsampler(
            aligned_clip_sam_feature_low=aligned_clip_sam_feature_low,
            sam3_feature_high=sam3_feature_high,
            clip_grid_hw=clip_grid_hw,
        )

        (
            clip_sam_feature_high,
            clip_coarse_logits,
            clip_coarse_pred,
        ) = self.clip_coarse_embedder(
            clip_image_feat_map_native=clip_image_feat_map_native,
            clip_text_tokens_native=clip_text_tokens_native,
            class_code=class_code,
            clip_sam_feature_high=clip_sam_feature_high,
            output_hw=(height, width),
        )

        mask_embed = self._build_initial_mask_embedding(
            semantic_logits=semantic_logits,
            class_code=class_code,
        )

        mask_logits_layers = []
        presence_logits_layers = []

        for layer in self.layers:
            (
                class_tokens,
                presence_logits,
                mask_embed,
            ) = layer(
                class_tokens=class_tokens,
                semantic_logits=semantic_logits,
                clip_sam_feature_high=clip_sam_feature_high,
                mask_embed=mask_embed,
                class_code=class_code,
            )

            mask_logits = self._build_mask_logits(
                mask_embed=mask_embed,
                class_code=class_code,
            )

            mask_logits_layers.append(mask_logits)
            presence_logits_layers.append(presence_logits)

        mask_logits_layers_tensor = torch.stack(mask_logits_layers, dim=0)
        presence_logits_layers_tensor = torch.stack(presence_logits_layers, dim=0)

        final_logits = mask_logits_layers_tensor[-1]
        presence_logits_last = presence_logits_layers_tensor[-1]

        if self.presence_enabled:
            presence_score = torch.sigmoid(presence_logits_last)
        else:
            presence_score = final_logits.new_ones(batch_size, num_classes)

        return {
            OUTPUT_KEYS.class_tokens: class_tokens.contiguous(),
            OUTPUT_KEYS.final_logits: final_logits.contiguous(),
            OUTPUT_KEYS.presence_logits: presence_logits_last.contiguous(),
            OUTPUT_KEYS.presence_score: presence_score.contiguous(),
            OUTPUT_KEYS.presence_logits_layers: presence_logits_layers_tensor.contiguous(),
            OUTPUT_KEYS.mask_logits_layers: mask_logits_layers_tensor.contiguous(),
            OUTPUT_KEYS.clip_coarse_logits: clip_coarse_logits.contiguous(),
            OUTPUT_KEYS.clip_coarse_pred: clip_coarse_pred.contiguous(),
        }