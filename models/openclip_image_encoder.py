from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn


class OpenCLIPImageEncoder(nn.Module):
    """
    Pure OpenCLIP vision wrapper.

    Responsibilities:
    1. Hold a loaded OpenCLIP visual tower.
    2. Expose a trunk-like interface for Sam3DualViTDetNeck.
    3. Default to returning the last low-resolution feature map in NCHW format.

    Non-responsibilities:
    - No channel projection
    - No external task-specific adaptation
    - No FPN / neck logic
    """

    def __init__(
        self,
        visual: nn.Module,
        default_output: str = "feat_map",
    ) -> None:
        super().__init__()
        self.visual = visual
        self.default_output = default_output

        feature_dim = self._infer_feature_dim(visual)
        self.channel_list = [feature_dim]

    @staticmethod
    def _infer_feature_dim(visual: nn.Module) -> int:
        candidates = [
            getattr(visual, "width", None),
            getattr(getattr(visual, "transformer", None), "width", None),
            getattr(visual, "num_features", None),
            getattr(visual, "embed_dim", None),
        ]
        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value
        raise AttributeError(
            "Cannot infer OpenCLIP visual feature dimension. "
            "Please inspect the visual tower and add a new rule here."
        )

    def _extract_last_tensor(self, obj: Any) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj

        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise ValueError("Received empty list/tuple while parsing image intermediates.")
            return self._extract_last_tensor(obj[-1])

        if isinstance(obj, dict):
            preferred_keys = (
                "image_intermediates",
                "intermediates",
                "features",
                "feature_maps",
                "x",
            )
            for key in preferred_keys:
                if key in obj:
                    return self._extract_last_tensor(obj[key])
            raise KeyError(
                f"Cannot find a known feature key in forward_intermediates output: {list(obj.keys())}"
            )

        raise TypeError(f"Unsupported intermediate output type: {type(obj)}")

    def _forward_intermediates(self, images: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.visual, "forward_intermediates"):
            raise RuntimeError(
                "The loaded OpenCLIP visual tower does not provide forward_intermediates(). "
                "Please use open-clip-torch>=2.32.0 and a compatible vision tower."
            )

        try:
            out = self.visual.forward_intermediates(
                images,
                indices=[-1],
                output_fmt="NCHW",
                intermediates_only=True,
            )
        except TypeError:
            # Fallback for slightly different implementations
            out = self.visual.forward_intermediates(images)

        feat_map = self._extract_last_tensor(out)

        if feat_map.ndim != 4:
            raise ValueError(
                f"Expected a 4D NCHW feature map, but got shape={tuple(feat_map.shape)}"
            )

        return feat_map

    def encode_image(
        self,
        images: torch.Tensor,
        output_mode: Optional[str] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        mode = output_mode or self.default_output

        feat_map = self._forward_intermediates(images)

        if mode == "feat_map":
            return feat_map

        if mode == "tokens":
            # [B, C, H, W] -> [B, HW, C]
            return feat_map.flatten(2).transpose(1, 2)

        if mode == "all":
            return {
                "feat_map": feat_map,
                "tokens": feat_map.flatten(2).transpose(1, 2),
            }

        raise ValueError(
            f"Unknown output_mode={mode}. "
            "Supported modes are: feat_map, tokens, all."
        )

    def forward(self, images: torch.Tensor):
        return self.encode_image(images, output_mode=self.default_output)