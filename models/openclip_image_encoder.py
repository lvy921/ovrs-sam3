from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenCLIPImageEncoder(nn.Module):
    """
    OpenCLIP vision wrapper.

    Design:
    1. Weights are loaded outside, in model_builder.py
    2. This wrapper only receives a ready-made visual tower
    3. It returns:
       - global image feature in CLIP space
       - patch tokens from the last 3rd block
       - patch tokens from the last 2nd block
    4. Positional embeddings are interpolated for larger inputs
    """

    def __init__(
        self,
        visual: nn.Module,
        default_output: str = "all",
    ) -> None:
        super().__init__()
        self.visual = visual
        self.default_output = default_output

        self._feature_dim = self._infer_feature_dim(visual)
        self.channel_list = [self._feature_dim]

    @property
    def output_dim(self) -> int:
        return int(self._feature_dim)

    @staticmethod
    def _infer_feature_dim(visual: nn.Module) -> int:
        # Prefer CLIP output-space dim if visual.proj exists
        proj = getattr(visual, "proj", None)
        if proj is not None and hasattr(proj, "shape") and len(proj.shape) == 2:
            return int(proj.shape[-1])

        candidates = [
            getattr(visual, "output_dim", None),
            getattr(visual, "width", None),
            getattr(getattr(visual, "transformer", None), "width", None),
            getattr(visual, "num_features", None),
            getattr(visual, "embed_dim", None),
        ]
        for value in candidates:
            if isinstance(value, int) and value > 0:
                return int(value)

        raise AttributeError(
            "Cannot infer OpenCLIP visual feature dimension."
        )

    @staticmethod
    def _to_2tuple(x) -> Tuple[int, int]:
        if isinstance(x, int):
            return (x, x)
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return (int(x[0]), int(x[1]))
        raise TypeError(f"Cannot convert to 2-tuple: {x!r}")

    def _get_base_grid_size(self) -> Tuple[int, int]:
        pos_embed = getattr(self.visual, "positional_embedding", None)
        if pos_embed is None:
            raise AttributeError("visual.positional_embedding is missing.")

        num_prefix_tokens = 1
        num_patch_tokens = int(pos_embed.shape[0]) - num_prefix_tokens
        if num_patch_tokens <= 0:
            raise ValueError(
                f"Invalid positional embedding shape: {tuple(pos_embed.shape)}"
            )

        grid_size = getattr(self.visual, "grid_size", None)
        if grid_size is not None:
            grid_h, grid_w = self._to_2tuple(grid_size)
            if grid_h * grid_w == num_patch_tokens:
                return grid_h, grid_w

        side = int(round(math.sqrt(num_patch_tokens)))
        if side * side != num_patch_tokens:
            raise ValueError(
                "Cannot infer a square base patch grid from positional embedding. "
                f"num_patch_tokens={num_patch_tokens}"
            )
        return side, side

    def _interpolate_positional_embedding(
        self,
        target_grid_hw: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        pos_embed = self.visual.positional_embedding
        if pos_embed.ndim != 2:
            raise ValueError(
                f"Expected visual.positional_embedding as [L, C], got {tuple(pos_embed.shape)}"
            )

        target_h, target_w = int(target_grid_hw[0]), int(target_grid_hw[1])
        base_h, base_w = self._get_base_grid_size()

        num_prefix_tokens = 1
        cls_pos = pos_embed[:num_prefix_tokens]   # [1, C]
        patch_pos = pos_embed[num_prefix_tokens:] # [H0*W0, C]
        embed_dim = int(patch_pos.shape[-1])

        if base_h == target_h and base_w == target_w:
            return pos_embed.to(device=device, dtype=dtype)

        patch_pos = patch_pos.reshape(base_h, base_w, embed_dim)
        patch_pos = patch_pos.permute(2, 0, 1).unsqueeze(0)  # [1, C, H0, W0]

        patch_pos = F.interpolate(
            patch_pos,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos = patch_pos.squeeze(0).permute(1, 2, 0).reshape(target_h * target_w, embed_dim)
        pos_embed_resized = torch.cat([cls_pos, patch_pos], dim=0)
        return pos_embed_resized.to(device=device, dtype=dtype)

    def _project_tokens_to_clip_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C] including cls token
        return: [B, L, D] in CLIP output space
        """
        if hasattr(self.visual, "ln_post") and self.visual.ln_post is not None:
            x = self.visual.ln_post(x)

        proj = getattr(self.visual, "proj", None)
        if proj is not None:
            x = x @ proj

        return x

    def _forward_vit_with_intermediate_tokens(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not all(hasattr(self.visual, name) for name in [
            "conv1",
            "class_embedding",
            "positional_embedding",
            "patch_dropout",
            "ln_pre",
            "transformer",
        ]):
            raise RuntimeError(
                "OpenCLIPImageEncoder currently only supports ViT-like visual towers."
            )

        x = self.visual.conv1(images)  # [B, C, Gh, Gw]
        if x.ndim != 4:
            raise ValueError(
                f"Expected conv1 output as [B, C, H, W], got {tuple(x.shape)}"
            )

        batch_size, width, grid_h, grid_w = x.shape

        x = x.reshape(batch_size, width, grid_h * grid_w).permute(0, 2, 1)  # [B, N, C]

        cls_token = self.visual.class_embedding.to(dtype=x.dtype, device=x.device)
        cls_token = cls_token.view(1, 1, -1).expand(batch_size, 1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+N, C]

        pos_embed = self._interpolate_positional_embedding(
            target_grid_hw=(grid_h, grid_w),
            dtype=x.dtype,
            device=x.device,
        )
        x = x + pos_embed.unsqueeze(0)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        # Transformer blocks
        x = x.permute(1, 0, 2)  # [L, B, C]

        resblocks = self.visual.transformer.resblocks
        num_blocks = len(resblocks)
        idx_lm3 = num_blocks - 3
        idx_lm2 = num_blocks - 2

        tokens_lm3 = None
        tokens_lm2 = None

        for i, block in enumerate(resblocks):
            x = block(x)
            if i == idx_lm3:
                tokens_lm3 = x.permute(1, 0, 2).contiguous()  # [B, L, C]
            if i == idx_lm2:
                tokens_lm2 = x.permute(1, 0, 2).contiguous()  # [B, L, C]

        x = x.permute(1, 0, 2).contiguous()  # [B, L, C]

        if tokens_lm3 is None or tokens_lm2 is None:
            raise RuntimeError(
                "Failed to capture tokens from last-3rd / last-2nd blocks."
            )

        # project to CLIP output space
        x = self._project_tokens_to_clip_space(x)
        tokens_lm3 = self._project_tokens_to_clip_space(tokens_lm3)
        tokens_lm2 = self._project_tokens_to_clip_space(tokens_lm2)

        image_feat = x[:, 0, :]             # [B, D]
        patch_tokens_lm3 = tokens_lm3[:, 1:, :]  # [B, N, D]
        patch_tokens_lm2 = tokens_lm2[:, 1:, :]  # [B, N, D]

        return {
            "image_feat": image_feat,
            "patch_tokens_lm3": patch_tokens_lm3,
            "patch_tokens_lm2": patch_tokens_lm2,
        }

    def encode_image(self, images: torch.Tensor, output_mode: str | None = None):
        mode = output_mode or self.default_output
        out = self._forward_vit_with_intermediate_tokens(images)

        if mode == "all":
            return out
        if mode == "image_feat":
            return out["image_feat"]
        if mode == "patch_tokens_lm3":
            return out["patch_tokens_lm3"]
        if mode == "patch_tokens_lm2":
            return out["patch_tokens_lm2"]

        raise ValueError(
            f"Unknown output_mode={mode}. "
            "Supported modes are: all, image_feat, patch_tokens_lm3, patch_tokens_lm2."
        )

    def forward(self, images: torch.Tensor):
        return self.encode_image(images, output_mode=self.default_output)