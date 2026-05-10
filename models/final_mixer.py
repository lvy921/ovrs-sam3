from __future__ import annotations

import math

import torch
import torch.nn as nn


def _choose_group_norm_groups(num_channels: int, preferred_groups: int = 8) -> int:
    for groups in (preferred_groups, 4, 2, 1):
        if groups <= num_channels and num_channels % groups == 0:
            return groups
    return 1


class SemanticScoreEmbedding(nn.Module):
    """
    Convert semantic logits into a small-channel spatial score embedding.

    Input:
        semantic_logits: [B, C, H, W]

    Output:
        score_embed: [B, C, score_dim, H, W]

    Symbol meanings:
        B means batch size.
        C means class count.
        H and W mean spatial height and width.
        score_dim means the small channel count used for spatial score state.
    """

    def __init__(
        self,
        score_dim: int = 32,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()

        self.score_dim = int(score_dim)
        if self.score_dim <= 0:
            raise ValueError(f"score_dim must be positive, got {score_dim}.")

        groups = _choose_group_norm_groups(
            num_channels=self.score_dim,
            preferred_groups=int(norm_groups),
        )

        self.net = nn.Sequential(
            nn.Conv2d(1, self.score_dim, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GroupNorm(groups, self.score_dim),
            nn.GELU(),
            nn.Conv2d(self.score_dim, self.score_dim, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(groups, self.score_dim),
            nn.GELU(),
        )

    def forward(self, semantic_logits: torch.Tensor) -> torch.Tensor:
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape

        x = semantic_logits.reshape(batch_size * num_classes, 1, height, width)
        x = self.net(x)

        return x.reshape(
            batch_size,
            num_classes,
            self.score_dim,
            height,
            width,
        ).contiguous()


class ClassPixelAttentionFusionLayer(nn.Module):
    """
    One fusion layer.

    It first lets class_query attend shared_clip_feature, then uses the updated
    class_query to guide per-pixel class-wise attention over score_embed.

    Inputs:
        score_embed:         [B, C, score_dim, H, W]
        class_query:         [B, C, sam_dim]
        shared_clip_feature: [B, N_clip, sam_dim]

    Outputs:
        updated_score_embed: [B, C, score_dim, H, W]
        updated_class_query: [B, C, sam_dim]
    """

    def __init__(
        self,
        sam_dim: int = 256,
        score_dim: int = 32,
        class_dim: int = 128,
        attn_dim: int = 160,
        num_heads: int = 8,
        dropout: float = 0.1,
        norm_groups: int = 8,
        use_ffn: bool = True,
    ) -> None:
        super().__init__()

        self.sam_dim = int(sam_dim)
        self.score_dim = int(score_dim)
        self.class_dim = int(class_dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.use_ffn = bool(use_ffn)

        if self.sam_dim <= 0:
            raise ValueError(f"sam_dim must be positive, got {sam_dim}.")
        if self.score_dim <= 0:
            raise ValueError(f"score_dim must be positive, got {score_dim}.")
        if self.class_dim <= 0:
            raise ValueError(f"class_dim must be positive, got {class_dim}.")
        if self.attn_dim <= 0:
            raise ValueError(f"attn_dim must be positive, got {attn_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.sam_dim % self.num_heads != 0:
            raise ValueError(
                "sam_dim must be divisible by num_heads, "
                f"got sam_dim={self.sam_dim}, num_heads={self.num_heads}."
            )
        if self.attn_dim % self.num_heads != 0:
            raise ValueError(
                "attn_dim must be divisible by num_heads, "
                f"got attn_dim={self.attn_dim}, num_heads={self.num_heads}."
            )
        if self.score_dim % self.num_heads != 0:
            raise ValueError(
                "score_dim must be divisible by num_heads for value heads, "
                f"got score_dim={self.score_dim}, num_heads={self.num_heads}."
            )

        self.class_memory_attn = nn.MultiheadAttention(
            embed_dim=self.sam_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.class_memory_norm = nn.LayerNorm(self.sam_dim)

        self.class_context_proj = nn.Linear(self.sam_dim, self.class_dim)

        pixel_in_dim = self.score_dim + self.class_dim
        self.q_proj = nn.Linear(pixel_in_dim, self.attn_dim)
        self.k_proj = nn.Linear(pixel_in_dim, self.attn_dim)
        self.v_proj = nn.Linear(self.score_dim, self.score_dim)
        self.out_proj = nn.Linear(self.score_dim, self.score_dim)

        self.dropout = nn.Dropout(float(dropout))

        groups = _choose_group_norm_groups(
            num_channels=self.score_dim,
            preferred_groups=int(norm_groups),
        )

        self.score_attn_norm = nn.GroupNorm(groups, self.score_dim)
        self.score_ffn_norm = nn.GroupNorm(groups, self.score_dim)

        if self.use_ffn:
            self.score_ffn = nn.Sequential(
                nn.Conv2d(self.score_dim, self.score_dim * 4, kernel_size=1),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Conv2d(self.score_dim * 4, self.score_dim, kernel_size=1),
            )
        else:
            self.score_ffn = None

    def _apply_score_norm(
        self,
        score_embed: torch.Tensor,
        norm: nn.GroupNorm,
    ) -> torch.Tensor:
        batch_size, num_classes, score_dim, height, width = score_embed.shape

        x = score_embed.reshape(
            batch_size * num_classes,
            score_dim,
            height,
            width,
        )
        x = norm(x)

        return x.reshape(
            batch_size,
            num_classes,
            score_dim,
            height,
            width,
        ).contiguous()

    def _run_class_pixel_attention(
        self,
        score_embed: torch.Tensor,
        class_query: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes, score_dim, height, width = score_embed.shape

        class_context = self.class_context_proj(class_query)
        class_context = class_context[:, :, :, None, None].expand(
            batch_size,
            num_classes,
            self.class_dim,
            height,
            width,
        )

        pixel_state = torch.cat([score_embed, class_context], dim=2)

        pixel_state = pixel_state.permute(0, 3, 4, 1, 2).contiguous()
        pixel_state = pixel_state.reshape(
            batch_size * height * width,
            num_classes,
            self.score_dim + self.class_dim,
        )

        score_state = score_embed.permute(0, 3, 4, 1, 2).contiguous()
        score_state = score_state.reshape(
            batch_size * height * width,
            num_classes,
            score_dim,
        )

        q = self.q_proj(pixel_state)
        k = self.k_proj(pixel_state)
        v = self.v_proj(score_state)

        q_head_dim = self.attn_dim // self.num_heads
        v_head_dim = self.score_dim // self.num_heads

        q = q.reshape(
            batch_size * height * width,
            num_classes,
            self.num_heads,
            q_head_dim,
        ).permute(0, 2, 1, 3)

        k = k.reshape(
            batch_size * height * width,
            num_classes,
            self.num_heads,
            q_head_dim,
        ).permute(0, 2, 1, 3)

        v = v.reshape(
            batch_size * height * width,
            num_classes,
            self.num_heads,
            v_head_dim,
        ).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(float(q_head_dim))

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(
            batch_size * height * width,
            num_classes,
            self.score_dim,
        )

        out = self.out_proj(out)

        out = out.reshape(
            batch_size,
            height,
            width,
            num_classes,
            self.score_dim,
        )
        out = out.permute(0, 3, 4, 1, 2).contiguous()

        return out

    def forward(
        self,
        score_embed: torch.Tensor,
        class_query: torch.Tensor,
        shared_clip_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if score_embed.dim() != 5:
            raise ValueError(
                "score_embed must be [B, C, score_dim, H, W], "
                f"got {tuple(score_embed.shape)}."
            )
        if class_query.dim() != 3:
            raise ValueError(
                "class_query must be [B, C, sam_dim], "
                f"got {tuple(class_query.shape)}."
            )
        if shared_clip_feature.dim() != 3:
            raise ValueError(
                "shared_clip_feature must be [B, N_clip, sam_dim], "
                f"got {tuple(shared_clip_feature.shape)}."
            )

        batch_size, num_classes, score_dim, _, _ = score_embed.shape

        if int(score_dim) != self.score_dim:
            raise ValueError(
                f"score_embed channel mismatch: expected {self.score_dim}, "
                f"got {score_dim}."
            )
        if tuple(class_query.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "class_query batch/class shape mismatch: "
                f"{tuple(class_query.shape[:2])} vs "
                f"{(batch_size, num_classes)}."
            )
        if tuple(shared_clip_feature.shape[:1]) != (batch_size,):
            raise ValueError(
                "shared_clip_feature batch shape mismatch: "
                f"{tuple(shared_clip_feature.shape[:1])} vs {(batch_size,)}."
            )
        if int(class_query.shape[-1]) != self.sam_dim:
            raise ValueError(
                f"class_query dim mismatch: expected {self.sam_dim}, "
                f"got {class_query.shape[-1]}."
            )
        if int(shared_clip_feature.shape[-1]) != self.sam_dim:
            raise ValueError(
                f"shared_clip_feature dim mismatch: expected {self.sam_dim}, "
                f"got {shared_clip_feature.shape[-1]}."
            )

        class_delta, _ = self.class_memory_attn(
            query=class_query,
            key=shared_clip_feature,
            value=shared_clip_feature,
            need_weights=False,
        )
        class_query = self.class_memory_norm(
            class_query + self.dropout(class_delta)
        )

        score_delta = self._run_class_pixel_attention(
            score_embed=score_embed,
            class_query=class_query,
        )
        score_embed = score_embed + self.dropout(score_delta)
        score_embed = self._apply_score_norm(
            score_embed=score_embed,
            norm=self.score_attn_norm,
        )

        if self.score_ffn is not None:
            batch_size, num_classes, score_dim, height, width = score_embed.shape
            ffn_in = self._apply_score_norm(
                score_embed=score_embed,
                norm=self.score_ffn_norm,
            )
            ffn_in = ffn_in.reshape(
                batch_size * num_classes,
                score_dim,
                height,
                width,
            )
            ffn_out = self.score_ffn(ffn_in)
            ffn_out = ffn_out.reshape(
                batch_size,
                num_classes,
                score_dim,
                height,
                width,
            )
            score_embed = score_embed + self.dropout(ffn_out)

        return score_embed.contiguous(), class_query.contiguous()


class ClassQuerySemanticFinalMixer(nn.Module):
    """
    New final mixer.

    Inputs:
        semantic_logits:     [B, C, H, W]
        class_query:         [B, C, sam_dim]
        shared_clip_feature: [B, N_clip, sam_dim]

    Output:
        final_logits: [B, C, H, W]
    """

    def __init__(
        self,
        sam_dim: int = 256,
        score_dim: int = 32,
        class_dim: int = 128,
        attn_dim: int = 160,
        num_heads: int = 8,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        use_final_residual: bool = True,
    ) -> None:
        super().__init__()

        self.sam_dim = int(sam_dim)
        self.score_dim = int(score_dim)
        self.class_dim = int(class_dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.fusion_layers = int(fusion_layers)
        self.use_final_residual = bool(use_final_residual)

        if self.fusion_layers <= 0:
            raise ValueError(
                f"fusion_layers must be positive, got {fusion_layers}."
            )

        self.score_embedding = SemanticScoreEmbedding(score_dim=self.score_dim)

        self.layers = nn.ModuleList(
            [
                ClassPixelAttentionFusionLayer(
                    sam_dim=self.sam_dim,
                    score_dim=self.score_dim,
                    class_dim=self.class_dim,
                    attn_dim=self.attn_dim,
                    num_heads=self.num_heads,
                    dropout=float(dropout),
                )
                for _ in range(self.fusion_layers)
            ]
        )

        groups = _choose_group_norm_groups(self.score_dim, preferred_groups=8)
        self.score_head = nn.Sequential(
            nn.Conv2d(self.score_dim, self.score_dim, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(groups, self.score_dim),
            nn.GELU(),
            nn.Conv2d(self.score_dim, 1, kernel_size=1, padding_mode="replicate"),
        )

    def forward(
        self,
        semantic_logits: torch.Tensor,
        class_query: torch.Tensor,
        shared_clip_feature: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )
        if class_query.dim() != 3:
            raise ValueError(
                "class_query must be [B, C, sam_dim], "
                f"got {tuple(class_query.shape)}."
            )
        if shared_clip_feature.dim() != 3:
            raise ValueError(
                "shared_clip_feature must be [B, N_clip, sam_dim], "
                f"got {tuple(shared_clip_feature.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape

        if tuple(class_query.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "class_query batch/class shape mismatch: "
                f"{tuple(class_query.shape[:2])} vs "
                f"{(batch_size, num_classes)}."
            )
        if int(class_query.shape[-1]) != self.sam_dim:
            raise ValueError(
                f"class_query dim mismatch: expected {self.sam_dim}, "
                f"got {class_query.shape[-1]}."
            )
        if int(shared_clip_feature.shape[0]) != int(batch_size):
            raise ValueError(
                "shared_clip_feature batch size mismatch: "
                f"{shared_clip_feature.shape[0]} vs {batch_size}."
            )
        if int(shared_clip_feature.shape[-1]) != self.sam_dim:
            raise ValueError(
                f"shared_clip_feature dim mismatch: expected {self.sam_dim}, "
                f"got {shared_clip_feature.shape[-1]}."
            )

        score_embed = self.score_embedding(semantic_logits)

        for layer in self.layers:
            score_embed, class_query = layer(
                score_embed=score_embed,
                class_query=class_query,
                shared_clip_feature=shared_clip_feature,
            )

        score_state = score_embed.reshape(
            batch_size * num_classes,
            self.score_dim,
            height,
            width,
        )
        delta_logits = self.score_head(score_state)
        delta_logits = delta_logits.reshape(
            batch_size,
            num_classes,
            height,
            width,
        )

        if self.use_final_residual:
            return semantic_logits + delta_logits

        return delta_logits