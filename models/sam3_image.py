from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_misc import BatchedDatapoint, FindStage
from .geometry_encoders import Prompt
from .task_modes import OUTPUT_KEYS, TASK_MODE_SEMANTIC, normalize_task_mode
from .vl_combiner import SAM3VLBackbone

def _choose_group_norm_groups(num_channels: int, preferred_groups: int = 8) -> int:
    for groups in (preferred_groups, 4, 2, 1):
        if groups <= num_channels and num_channels % groups == 0:
            return groups
    return 1


class BranchEmbeddingScoreFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        norm_groups: int = 8,
        class_attn_heads: int = 4,
        class_attn_pooling_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.class_attn_heads = int(class_attn_heads)
        self.class_attn_pooling_size = int(class_attn_pooling_size)

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if self.class_attn_heads <= 0:
            raise ValueError(
                f"class_attn_heads must be positive, got {class_attn_heads}."
            )
        if self.hidden_dim % self.class_attn_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by class_attn_heads, "
                f"got hidden_dim={self.hidden_dim}, "
                f"class_attn_heads={self.class_attn_heads}."
            )

        groups = _choose_group_norm_groups(
            num_channels=self.hidden_dim,
            preferred_groups=int(norm_groups),
        )

        self.sam_score_embed = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )

        self.clip_score_embed = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                2 * self.hidden_dim,
                self.hidden_dim,
                kernel_size=7,
                padding=3,
            ),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
        )

        self.class_attn_norm = nn.LayerNorm(self.hidden_dim)
        self.class_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.class_attn_heads,
            dropout=float(dropout),
            batch_first=True,
        )

        self.class_ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.class_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

    def _pool_fused_memory(
        self,
        fused_memory: torch.Tensor,
    ) -> torch.Tensor:
        pooling_size = int(self.class_attn_pooling_size)
        if pooling_size <= 1:
            return fused_memory

        batch_size, num_classes, hidden_dim, height, width = fused_memory.shape

        if height < pooling_size or width < pooling_size:
            return fused_memory

        x = fused_memory.reshape(
            batch_size * num_classes,
            hidden_dim,
            height,
            width,
        )

        x = F.avg_pool2d(
            x,
            kernel_size=pooling_size,
            stride=pooling_size,
        )

        pooled_h, pooled_w = int(x.shape[-2]), int(x.shape[-1])
        return x.reshape(
            batch_size,
            num_classes,
            hidden_dim,
            pooled_h,
            pooled_w,
        )

    def _run_pooled_class_attention(
        self,
        fused_memory: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, num_classes, hidden_dim, pooled_h, pooled_w = fused_memory.shape

        x = fused_memory.permute(0, 3, 4, 1, 2).contiguous()
        x = x.reshape(
            batch_size * pooled_h * pooled_w,
            num_classes,
            hidden_dim,
        )

        attn_in = self.class_attn_norm(x)
        attn_out, _ = self.class_attn(
            query=attn_in,
            key=attn_in,
            value=attn_in,
            need_weights=False,
        )
        x = x + attn_out

        ffn_in = self.class_ffn_norm(x)
        x = x + self.class_ffn(ffn_in)

        x = x.reshape(
            batch_size,
            pooled_h,
            pooled_w,
            num_classes,
            hidden_dim,
        )
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if (pooled_h, pooled_w) == (target_h, target_w):
            return x

        x = x.reshape(
            batch_size * num_classes,
            hidden_dim,
            pooled_h,
            pooled_w,
        )
        x = F.interpolate(
            x,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return x.reshape(
            batch_size,
            num_classes,
            hidden_dim,
            target_h,
            target_w,
        )

    def forward(
        self,
        semantic_logits: torch.Tensor,
        clip_dense_logits: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_logits.dim() != 4:
            raise ValueError(
                f"semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )
        if clip_dense_logits.dim() != 4:
            raise ValueError(
                f"clip_dense_logits must be [B, C, H, W], "
                f"got {tuple(clip_dense_logits.shape)}."
            )
        if semantic_logits.shape != clip_dense_logits.shape:
            raise ValueError(
                "semantic_logits and clip_dense_logits must have the same shape, "
                f"got {tuple(semantic_logits.shape)} and "
                f"{tuple(clip_dense_logits.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape

        sam_input = semantic_logits.reshape(batch_size * num_classes, 1, height, width)
        clip_input = clip_dense_logits.reshape(batch_size * num_classes, 1, height, width)

        sam_embed = self.sam_score_embed(sam_input)
        clip_embed = self.clip_score_embed(clip_input)

        fused = torch.cat([sam_embed, clip_embed], dim=1)
        fused = self.fusion_conv(fused)

        # Anchor the fused feature to SAM3 boundary-aware score features.
        fused = fused + sam_embed

        fused_memory = fused.reshape(
            batch_size,
            num_classes,
            self.hidden_dim,
            height,
            width,
        )

        pooled_memory = self._pool_fused_memory(fused_memory)
        class_delta = self._run_pooled_class_attention(
            fused_memory=pooled_memory,
            target_hw=(height, width),
        )

        fused_memory = fused_memory + class_delta

        if fused_memory.shape[:2] != semantic_logits.shape[:2]:
            raise RuntimeError(
                "fused_memory batch/class shape mismatch: "
                f"{tuple(fused_memory.shape[:2])} vs "
                f"{tuple(semantic_logits.shape[:2])}."
            )
        if fused_memory.shape[-2:] != semantic_logits.shape[-2:]:
            raise RuntimeError(
                "fused_memory spatial shape mismatch: "
                f"{tuple(fused_memory.shape[-2:])} vs "
                f"{tuple(semantic_logits.shape[-2:])}."
            )

        return fused_memory

class Sam3Image(torch.nn.Module):
    def __init__(
        self,
        backbone: SAM3VLBackbone,
        transformer,
        input_geometry_encoder,
        segmentation_head=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multimask_output: bool = True,
        use_act_checkpoint_seg_head: bool = True,
        interactivity_in_encoder: bool = True,
        matcher=None,
        use_dot_prod_scoring=True,
        supervise_joint_box_scores: bool = False,
        detach_presence_in_joint_score: bool = False,
        separate_scorer_for_instance: bool = False,
        num_interactive_steps_val: int = 0,
        clip_image_encoder=None,
        clip_text_encoder=None,
        openclip_cfg=None,
        final_mixer_cfg=None,
        task_mode: str = TASK_MODE_SEMANTIC,
        **kwargs,
    ):
        super().__init__()

        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head

        # Kept for build/config compatibility.
        self.o2m_mask_predict = o2m_mask_predict
        self.dot_prod_scoring = dot_prod_scoring
        self.use_act_checkpoint_seg_head = use_act_checkpoint_seg_head
        self.interactivity_in_encoder = interactivity_in_encoder
        self.matcher = matcher
        self.num_interactive_steps_val = num_interactive_steps_val
        self.use_dot_prod_scoring = use_dot_prod_scoring

        self.clip_image_encoder = clip_image_encoder
        self.clip_text_encoder = clip_text_encoder

        self.register_buffer(
            "openclip_image_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "openclip_image_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.task_mode = normalize_task_mode(task_mode)
        if self.task_mode != TASK_MODE_SEMANTIC:
            raise NotImplementedError("Sam3Image currently only supports semantic task mode.")

        self.clip_extra_token_templates: List[str] = []
        self.num_clip_extra_tokens = 0
        self.normalize_label_for_clip = True
        if openclip_cfg is not None:
            self.clip_extra_token_templates = list(getattr(openclip_cfg, "extra_token_templates", []))
            self.num_clip_extra_tokens = int(
                getattr(openclip_cfg, "num_extra_tokens", len(self.clip_extra_token_templates))
            )
            self.clip_extra_token_templates = self.clip_extra_token_templates[: self.num_clip_extra_tokens]
            self.normalize_label_for_clip = bool(getattr(openclip_cfg, "normalize_label_for_clip", True))

        if (self.clip_text_encoder is None) != (self.clip_image_encoder is None):
            raise RuntimeError(
                "OpenCLIP is partially initialized: clip_text_encoder and "
                "clip_image_encoder must either both exist or both be None."
            )

        self.clip_text_dim = self._infer_clip_text_dim() if self.clip_text_encoder is not None else None
        self.clip_image_dim = self._infer_clip_image_dim() if self.clip_image_encoder is not None else None
        self.clip_align_dim = None

        if self.clip_text_dim is not None and self.clip_image_dim is not None:
            if self.clip_text_dim != self.clip_image_dim:
                raise ValueError(
                    "Projected OpenCLIP text/image dimensions must match for native CLIP attention. "
                    f"Got text_dim={self.clip_text_dim}, image_dim={self.clip_image_dim}."
                )
            self.clip_align_dim = self.clip_text_dim

        self.clip_native_image_to_text_attn = None
        self.clip_native_image_to_text_norm = None
        self.clip_hv_proj = None
        self.sam3_to_clip_hv_attn = None
        self.sam3_to_clip_hv_norm = None
        self.extra_type_embed = None
        self.extra_token_mask_query_proj = None
        self.extra_token_mask_memory_proj = None
        self.extra_token_logit_scale = None

        if self.clip_align_dim is not None:
            self.clip_native_image_to_text_attn = nn.MultiheadAttention(
                embed_dim=self.clip_align_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.clip_native_image_to_text_norm = nn.LayerNorm(self.clip_align_dim)
            self.clip_hv_proj = nn.Linear(self.clip_align_dim, self.hidden_dim)

            self.sam3_to_clip_hv_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.sam3_to_clip_hv_norm = nn.LayerNorm(self.hidden_dim)
            self.extra_type_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

            self.extra_token_mask_query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.extra_token_mask_memory_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.extra_token_logit_scale = nn.Parameter(torch.tensor(2.3, dtype=torch.float32))

        self.final_mixer_hidden_dim = int(
            getattr(final_mixer_cfg, "hidden_dim", 128)
        )
        self.final_mixer_num_heads = int(
            getattr(final_mixer_cfg, "num_heads", 8)
        )
        self.final_mixer_dropout = float(
            getattr(final_mixer_cfg, "dropout", 0.1)
        )
        gate_bias_init = float(
            getattr(final_mixer_cfg, "gate_bias_init", 3.0)
        )
        self.final_mixer_class_attn_heads = int(
            getattr(final_mixer_cfg, "class_attn_heads", 4)
        )
        self.final_mixer_class_attn_pooling_size = int(
            getattr(final_mixer_cfg, "class_attn_pooling_size", 2)
        )

        if self.final_mixer_hidden_dim % self.final_mixer_num_heads != 0:
            raise ValueError(
                "final_mixer_hidden_dim must be divisible by final_mixer_num_heads, "
                f"got hidden_dim={self.final_mixer_hidden_dim}, "
                f"num_heads={self.final_mixer_num_heads}."
            )

        self.class_query_seed_proj = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        nn.init.xavier_uniform_(self.class_query_seed_proj.weight)
        nn.init.zeros_(self.class_query_seed_proj.bias)

        self.class_query_encoder_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.class_query_encoder_cross_attn_norm = nn.LayerNorm(self.hidden_dim)

        if self.final_mixer_hidden_dim == self.hidden_dim:
            self.class_query_proj = nn.Identity()
        else:
            self.class_query_proj = nn.Linear(self.hidden_dim, self.final_mixer_hidden_dim)

        self.class_query_self_attn = nn.MultiheadAttention(
            embed_dim=self.final_mixer_hidden_dim,
            num_heads=self.final_mixer_num_heads,
            dropout=self.final_mixer_dropout,
            batch_first=True,
        )
        self.class_query_self_attn_norm = nn.LayerNorm(self.final_mixer_hidden_dim)

        self.score_fusion_block = BranchEmbeddingScoreFusion(
            hidden_dim=self.final_mixer_hidden_dim,
            norm_groups=8,
            class_attn_heads=self.final_mixer_class_attn_heads,
            class_attn_pooling_size=self.final_mixer_class_attn_pooling_size,
            dropout=self.final_mixer_dropout,
        )

        self.suppression_query_cross_attn = nn.MultiheadAttention(
            embed_dim=self.final_mixer_hidden_dim,
            num_heads=self.final_mixer_num_heads,
            dropout=self.final_mixer_dropout,
            batch_first=True,
        )
        self.suppression_query_cross_attn_norm = nn.LayerNorm(self.final_mixer_hidden_dim)

        self.suppression_logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.suppression_gate_bias = nn.Parameter(torch.tensor(gate_bias_init, dtype=torch.float32))

        self.prompt_chunk_size = None
        self._text_cache: Optional[Dict[str, torch.Tensor]] = None
        self._text_cache_key: Optional[Tuple[str, ...]] = None
        self._text_cache_device: Optional[str] = None

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        self._device = None
        self.clear_text_cache()
        return super().to(*args, **kwargs)

    @staticmethod
    def _normalize_text_cache_key(class_texts: List[str]) -> Tuple[str, ...]:
        return tuple(str(x) for x in class_texts)

    def clear_text_cache(self) -> None:
        self._text_cache = None
        self._text_cache_key = None
        self._text_cache_device = None

    def prepare_text_cache(
        self,
        class_texts: List[str],
        device: Optional[torch.device] = None,
        force: bool = False,
    ) -> None:
        if len(class_texts) == 0:
            raise ValueError("class_texts is empty, cannot build text cache.")

        device = torch.device(device) if device is not None else self.device
        cache_key = self._normalize_text_cache_key(class_texts)
        cache_device = str(device)

        if (
            not force
            and self._text_cache is not None
            and self._text_cache_key == cache_key
            and self._text_cache_device == cache_device
        ):
            return

        with torch.no_grad():
            text_out = self.backbone.forward_text(class_texts, device=device)
        text_out = self._detach_tree(text_out)

        cache: Dict[str, torch.Tensor] = {
            "language_features": text_out["language_features"].contiguous(),
            "language_mask": text_out["language_mask"].contiguous(),
        }
        if text_out.get("language_embeds") is not None:
            cache["language_embeds"] = text_out["language_embeds"].contiguous()

        if self.clip_text_encoder is not None and len(self.clip_extra_token_templates) > 0:
            with torch.no_grad():
                clip_text_tokens = self.clip_text_encoder.encode_prompt_templates(
                    class_names=class_texts,
                    templates=self.clip_extra_token_templates,
                    device=device,
                    normalize_label=self.normalize_label_for_clip,
                )
            cache["clip_text_tokens_native"] = clip_text_tokens.detach().contiguous()

        self._text_cache = cache
        self._text_cache_key = cache_key
        self._text_cache_device = cache_device

    def ensure_text_cache(self, class_texts: List[str], device: Optional[torch.device] = None) -> None:
        self.prepare_text_cache(class_texts=class_texts, device=device, force=False)

    def _slice_text_cache(self, start: int, end: int) -> Dict[str, torch.Tensor]:
        if self._text_cache is None:
            raise RuntimeError("Text cache is not prepared.")

        out = {
            "language_features": self._text_cache["language_features"][:, start:end].contiguous(),
            "language_mask": self._text_cache["language_mask"][start:end].contiguous(),
        }
        for key in ("language_embeds", "clip_text_tokens_native"):
            if key in self._text_cache:
                out[key] = self._text_cache[key][start:end].contiguous() if key == "clip_text_tokens_native" else self._text_cache[key][:, start:end].contiguous()
        return out

    def _get_prompt_chunk_size(self, num_classes: int) -> int:
        chunk_size = getattr(self, "prompt_chunk_size", None)
        if chunk_size is None or int(chunk_size) <= 0:
            return num_classes
        return min(int(chunk_size), num_classes)

    def _detach_tree(self, obj: Any):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        if isinstance(obj, dict):
            return {k: self._detach_tree(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._detach_tree(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._detach_tree(v) for v in obj)
        return obj

    def _infer_clip_text_dim(self) -> int:
        output_dim = getattr(self.clip_text_encoder, "output_dim", None)
        if isinstance(output_dim, int) and output_dim > 0:
            return output_dim
        raise AttributeError("clip_text_encoder must expose a positive integer `output_dim`.")

    def _infer_clip_image_dim(self) -> int:
        output_dim = getattr(self.clip_image_encoder, "output_dim", None)
        if isinstance(output_dim, int) and output_dim > 0:
            return output_dim
        raise AttributeError("clip_image_encoder must expose a positive integer `output_dim`.")

    def _get_openclip_patch_size(self) -> Tuple[int, int]:
        visual = self.clip_image_encoder.visual
        patch_size = getattr(visual, "patch_size", None)
        if isinstance(patch_size, int):
            return (patch_size, patch_size)
        if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            return (int(patch_size[0]), int(patch_size[1]))

        conv1 = getattr(visual, "conv1", None)
        kernel_size = getattr(conv1, "kernel_size", None) if conv1 is not None else None
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            return (int(kernel_size[0]), int(kernel_size[1]))

        raise AttributeError("Cannot infer OpenCLIP patch size.")

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        return int(value) if multiple <= 1 else ((int(value) + multiple - 1) // multiple) * multiple

    @staticmethod
    def _pad_chw_image(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        pad_h = max(0, int(out_h) - int(x.shape[-2]))
        pad_w = max(0, int(out_w) - int(x.shape[-1]))
        return x if pad_h == 0 and pad_w == 0 else F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

    def _prepare_openclip_image_batch(self, raw_images: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        if len(raw_images) == 0:
            raise ValueError("raw_images is empty.")

        processed = []
        for i, x in enumerate(raw_images):
            if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] != 3:
                raise ValueError(
                    f"raw_images[{i}] must be a tensor with shape [3, H, W], got "
                    f"{None if not isinstance(x, torch.Tensor) else tuple(x.shape)}"
                )
            processed.append(x.to(device=device, dtype=torch.float32))

        patch_h, patch_w = self._get_openclip_patch_size()
        max_h = self._round_up_to_multiple(max(int(x.shape[-2]) for x in processed), patch_h)
        max_w = self._round_up_to_multiple(max(int(x.shape[-1]) for x in processed), patch_w)

        batch = torch.stack([self._pad_chw_image(x, max_h, max_w) for x in processed], dim=0)
        return (batch - self.openclip_image_mean) / self.openclip_image_std

    def _build_clip_image_cache(
        self,
        input: BatchedDatapoint,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.clip_image_encoder is None:
            return None
        if input.raw_images is None:
            raise ValueError("clip_image_encoder is enabled, but BatchedDatapoint.raw_images is None.")

        clip_img_batch = self._prepare_openclip_image_batch(raw_images=input.raw_images, device=device)
        with torch.no_grad():
            clip_feat_map = self.clip_image_encoder(clip_img_batch)

        if not isinstance(clip_feat_map, torch.Tensor) or clip_feat_map.ndim != 4:
            raise ValueError(
                "clip_image_encoder must return a tensor with shape [B, D_clip, Hc, Wc]."
            )

        clip_feat_map = clip_feat_map.detach().contiguous()
        return {
            "clip_image_feat_map_native": clip_feat_map,
            "clip_image_grid_hw": (int(clip_feat_map.shape[-2]), int(clip_feat_map.shape[-1])),
        }

    def _expand_sam3_text_to_pairs(
        self,
        sam3_text_feats: torch.Tensor,
        sam3_text_mask: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # sam3_text_feats: [M, C, D], sam3_text_mask: [C, M]
        seq_len, num_classes, dim = sam3_text_feats.shape
        feats = sam3_text_feats.permute(1, 0, 2).contiguous()
        feats = feats.unsqueeze(0).expand(batch_size, num_classes, seq_len, dim)
        feats = feats.reshape(batch_size * num_classes, seq_len, dim).contiguous()

        mask = sam3_text_mask.unsqueeze(0).expand(batch_size, num_classes, seq_len)
        mask = mask.reshape(batch_size * num_classes, seq_len).contiguous()
        return feats, mask

    def _build_clip_hv_native(
            self,
            clip_image_feat_map_native: torch.Tensor,
            clip_text_tokens_native: torch.Tensor,
    ) -> torch.Tensor:
        if self.clip_native_image_to_text_attn is None or self.clip_native_image_to_text_norm is None:
            raise RuntimeError("OpenCLIP native attention modules are not initialized.")

        batch_size, image_dim, grid_h, grid_w = clip_image_feat_map_native.shape
        num_classes, num_templates, text_dim = clip_text_tokens_native.shape

        if image_dim != text_dim:
            raise ValueError(f"CLIP image/text dim mismatch: {image_dim} vs {text_dim}.")

        device = clip_image_feat_map_native.device
        dtype = clip_image_feat_map_native.dtype
        clip_text_tokens_native = clip_text_tokens_native.to(device=device, dtype=dtype)

        num_clip_tokens = int(grid_h) * int(grid_w)

        # [B, D_clip, Hc, Wc] -> [B, N_clip, D_clip]
        image_tokens = clip_image_feat_map_native.flatten(2).transpose(1, 2).contiguous()

        image_tokens = F.normalize(image_tokens, dim=-1, eps=1e-6)
        clip_text_tokens_native = F.normalize(
            clip_text_tokens_native,
            dim=-1,
            eps=1e-6,
        )

        # [B, N_clip, D_clip] -> [B*C_chunk, N_clip, D_clip]
        query = image_tokens[:, None].expand(
            batch_size,
            num_classes,
            num_clip_tokens,
            image_dim,
        )
        query = query.reshape(
            batch_size * num_classes,
            num_clip_tokens,
            image_dim,
        ).contiguous()

        # [C_chunk, K, D_clip] -> [B*C_chunk, K, D_clip]
        key_value = clip_text_tokens_native[None].expand(
            batch_size,
            num_classes,
            num_templates,
            text_dim,
        )
        key_value = key_value.reshape(
            batch_size * num_classes,
            num_templates,
            text_dim,
        ).contiguous()

        text_message, _ = self.clip_native_image_to_text_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )

        hv_pair = self.clip_native_image_to_text_norm(query + text_message)

        return hv_pair.reshape(
            batch_size,
            num_classes,
            num_clip_tokens,
            image_dim,
        ).contiguous()

    def _build_extra_tokens_from_clip_hv(
        self,
        hv_native: torch.Tensor,
        sam3_text_tokens: torch.Tensor,
        sam3_text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            self.clip_hv_proj is None
            or self.sam3_to_clip_hv_attn is None
            or self.sam3_to_clip_hv_norm is None
            or self.extra_type_embed is None
        ):
            raise RuntimeError("OpenCLIP extra token modules are not initialized.")

        batch_size, num_classes, num_clip_tokens, _ = hv_native.shape
        pair_count = batch_size * num_classes
        if sam3_text_tokens.shape[0] != pair_count:
            raise ValueError(
                f"Pair count mismatch: hv_native has {pair_count}, "
                f"sam3_text_tokens has {sam3_text_tokens.shape[0]}."
            )

        hv_native = hv_native.to(device=sam3_text_tokens.device, dtype=sam3_text_tokens.dtype)
        hv_pair = self.clip_hv_proj(hv_native).reshape(pair_count, num_clip_tokens, self.hidden_dim).contiguous()

        extra_delta, _ = self.sam3_to_clip_hv_attn(
            query=sam3_text_tokens,
            key=hv_pair,
            value=hv_pair,
            need_weights=False,
        )
        extra_tokens = self.sam3_to_clip_hv_norm(sam3_text_tokens + extra_delta)
        return extra_tokens + self.extra_type_embed.to(device=extra_tokens.device, dtype=extra_tokens.dtype)

    def _build_extra_token_aux_logits(
        self,
        extra_tokens: torch.Tensor,
        extra_token_mask: torch.Tensor,
        encoder_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if (
            self.extra_token_mask_query_proj is None
            or self.extra_token_mask_memory_proj is None
            or self.extra_token_logit_scale is None
        ):
            raise RuntimeError("extra token auxiliary mask modules are not initialized.")

        num_pairs = int(extra_tokens.shape[0])
        encoder_tokens, encoder_padding_mask = self._prepare_encoder_tokens(
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            padding_mask=encoder_out.get("padding_mask", None),
            num_pairs=num_pairs,
        )
        if encoder_padding_mask is not None and encoder_padding_mask.any():
            raise ValueError("extra_token_aux_logits expects encoder memory without padded spatial tokens.")

        spatial_shapes = encoder_out["spatial_shapes"]
        if isinstance(spatial_shapes, torch.Tensor):
            enc_h, enc_w = int(spatial_shapes[0, 0].item()), int(spatial_shapes[0, 1].item())
        else:
            enc_h, enc_w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])

        if encoder_tokens.shape[1] != enc_h * enc_w:
            raise ValueError(
                f"Encoder token count does not match spatial shape: "
                f"{encoder_tokens.shape[1]} vs {enc_h * enc_w}."
            )

        extra_class_token = self._masked_mean_pool(extra_tokens, extra_token_mask)
        q = F.normalize(self.extra_token_mask_query_proj(extra_class_token), dim=-1)
        k = F.normalize(self.extra_token_mask_memory_proj(encoder_tokens), dim=-1)
        aux_logits = torch.einsum("bd,bsd->bs", q, k) * self.extra_token_logit_scale.exp().clamp(max=100.0)
        return aux_logits.reshape(num_pairs, enc_h, enc_w).contiguous()

    @staticmethod
    def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).to(dtype=x.dtype).unsqueeze(-1)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def _build_class_query_seed(
            self,
            extra_summary: torch.Tensor,
            sam3_summary: torch.Tensor,
    ) -> torch.Tensor:
        query_seed = torch.cat(
            [sam3_summary, extra_summary, sam3_summary * extra_summary],
            dim=-1,
        )
        return self.class_query_seed_proj(query_seed)

    @staticmethod
    def _prepare_encoder_tokens(
        encoder_hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        num_pairs: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states.shape[0] == num_pairs:
            encoder_tokens = encoder_hidden_states.contiguous()
        elif encoder_hidden_states.shape[1] == num_pairs:
            encoder_tokens = encoder_hidden_states.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "Cannot infer encoder token layout: "
                f"encoder_hidden_states.shape={tuple(encoder_hidden_states.shape)}, num_pairs={num_pairs}."
            )

        if padding_mask is not None:
            if padding_mask.shape[0] != num_pairs or padding_mask.shape[1] != encoder_tokens.shape[1]:
                raise ValueError(
                    f"padding_mask shape mismatch: expected "
                    f"({num_pairs}, {encoder_tokens.shape[1]}), got {tuple(padding_mask.shape)}."
                )
            padding_mask = padding_mask.contiguous()

        return encoder_tokens, padding_mask

    def _run_class_query_encoder_cross_attn(
            self,
            class_query_seed: torch.Tensor,
            encoder_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        num_pairs = int(class_query_seed.shape[0])

        encoder_tokens, encoder_padding_mask = self._prepare_encoder_tokens(
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            padding_mask=encoder_out.get("padding_mask", None),
            num_pairs=num_pairs,
        )

        encoder_tokens = encoder_tokens.detach()
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.detach()

        query = class_query_seed.unsqueeze(1)

        attn_out, _ = self.class_query_encoder_cross_attn(
            query=query,
            key=encoder_tokens,
            value=encoder_tokens,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
        )

        class_query = self.class_query_encoder_cross_attn_norm(query + attn_out)
        return class_query.squeeze(1)

    def _build_clip_dense_logits(
            self,
            clip_image_feat_map_native: torch.Tensor,
            clip_text_tokens_native: torch.Tensor,
            target_hw: tuple[int, int],
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

        batch_size, image_dim, _, _ = clip_image_feat_map_native.shape
        num_classes, _, text_dim = clip_text_tokens_native.shape

        if image_dim != text_dim:
            raise ValueError(
                f"CLIP image/text dim mismatch: image_dim={image_dim}, text_dim={text_dim}."
            )

        clip_text_tokens_native = clip_text_tokens_native.to(
            device=clip_image_feat_map_native.device,
            dtype=clip_image_feat_map_native.dtype,
        )

        image_feat = F.normalize(clip_image_feat_map_native, dim=1, eps=1e-6)
        text_feat = F.normalize(clip_text_tokens_native, dim=-1, eps=1e-6)

        class_text_feat = text_feat.mean(dim=1)
        class_text_feat = F.normalize(class_text_feat, dim=-1, eps=1e-6)

        clip_dense_logits = torch.einsum(
            "bdhw,cd->bchw",
            image_feat,
            class_text_feat,
        )

        if tuple(clip_dense_logits.shape[-2:]) != tuple(target_hw):
            clip_dense_logits = F.interpolate(
                clip_dense_logits,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        if clip_dense_logits.shape[0] != batch_size:
            raise ValueError("Internal error: CLIP dense logits batch size changed.")
        if clip_dense_logits.shape[1] != num_classes:
            raise ValueError("Internal error: CLIP dense logits class count changed.")

        return clip_dense_logits.contiguous()

    def iter_chunk_raw_outputs(self, input: BatchedDatapoint) -> Iterator[Dict[str, Any]]:
        device = self.device

        if len(input.find_inputs) != 1:
            raise ValueError("Current semantic-only pipeline assumes exactly one find stage per batch.")

        base_find_input = input.find_inputs[0]
        class_texts = list(input.find_text_batch)
        if len(class_texts) == 0:
            raise ValueError("find_text_batch is empty.")

        self.ensure_text_cache(class_texts=class_texts, device=device)

        batch_size = int(input.img_batch.shape[0])
        num_classes = len(class_texts)
        chunk_size = self._get_prompt_chunk_size(num_classes)

        with torch.no_grad():
            image_backbone_out = self.backbone.forward_image(input.img_batch)
        image_backbone_out = self._detach_tree(image_backbone_out)

        clip_image_cache = self._build_clip_image_cache(input=input, device=device)
        if clip_image_cache is None:
            raise ValueError("clip_image_cache is None. Current semantic pipeline expects OpenCLIP image features.")

        for start in range(0, num_classes, chunk_size):
            end = min(start + chunk_size, num_classes)
            chunk_texts = class_texts[start:end]
            num_chunk_classes = len(chunk_texts)
            chunk_class_ids = list(range(start, end))
            chunk_text_cache = self._slice_text_cache(start=start, end=end)

            chunk_backbone_out = dict(image_backbone_out)
            chunk_backbone_out["language_features"] = chunk_text_cache["language_features"]
            chunk_backbone_out["language_mask"] = chunk_text_cache["language_mask"]
            if "language_embeds" in chunk_text_cache:
                chunk_backbone_out["language_embeds"] = chunk_text_cache["language_embeds"]

            chunk_find_input = self._build_prompt_expanded_find_stage(
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
                device=device,
                base_find_input=base_find_input,
            )

            sam3_pair_feats, sam3_pair_mask = self._expand_sam3_text_to_pairs(
                sam3_text_feats=chunk_backbone_out["language_features"],
                sam3_text_mask=chunk_backbone_out["language_mask"],
                batch_size=batch_size,
            )

            if "clip_text_tokens_native" not in chunk_text_cache:
                raise ValueError("clip_text_tokens_native is missing. Check openclip_cfg.extra_token_templates.")

            hv_native = self._build_clip_hv_native(
                clip_image_feat_map_native=clip_image_cache["clip_image_feat_map_native"],
                clip_text_tokens_native=chunk_text_cache["clip_text_tokens_native"],
            )
            extra_tokens = self._build_extra_tokens_from_clip_hv(
                hv_native=hv_native,
                sam3_text_tokens=sam3_pair_feats,
                sam3_text_mask=sam3_pair_mask,
            )

            extra_summary = self._masked_mean_pool(extra_tokens, sam3_pair_mask)
            sam3_summary = self._masked_mean_pool(sam3_pair_feats, sam3_pair_mask)

            chunk_backbone_out["clip_language_features_pair"] = extra_tokens.transpose(0, 1).contiguous()
            chunk_backbone_out["clip_language_mask_pair"] = sam3_pair_mask
            chunk_backbone_out["class_query_seed_pair"] = self._build_class_query_seed(
                extra_summary=extra_summary.detach(),
                sam3_summary=sam3_summary.detach(),
            )
            chunk_backbone_out["extra_tokens_pair"] = extra_tokens
            chunk_backbone_out["extra_tokens_mask_pair"] = sam3_pair_mask

            geometric_prompt = Prompt(
                box_embeddings=chunk_find_input.input_boxes,
                box_mask=chunk_find_input.input_boxes_mask,
                box_labels=chunk_find_input.input_boxes_label,
            )

            raw_outputs = self.forward_grounding_raw(
                backbone_out=chunk_backbone_out,
                find_input=chunk_find_input,
                geometric_prompt=geometric_prompt,
            )

            chunk_outputs = self._extract_and_reshape_chunk_outputs(
                raw_outputs=raw_outputs,
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
            )

            semantic_logits = self._ensure_4d_logits(
                chunk_outputs[OUTPUT_KEYS.semantic_logits],
                OUTPUT_KEYS.semantic_logits,
            )

            chunk_outputs[OUTPUT_KEYS.semantic_logits] = semantic_logits
            chunk_outputs[OUTPUT_KEYS.clip_dense_logits] = self._build_clip_dense_logits(
                clip_image_feat_map_native=clip_image_cache["clip_image_feat_map_native"],
                clip_text_tokens_native=chunk_text_cache["clip_text_tokens_native"],
                target_hw=tuple(semantic_logits.shape[-2:]),
            )

            yield {
                "chunk_start": start,
                "chunk_end": end,
                "chunk_class_ids": chunk_class_ids,
                "chunk_class_names": chunk_texts,
                "raw_outputs": chunk_outputs,
            }

    @staticmethod
    def _has_nonempty_geometric_prompt(find_input: Optional[FindStage]) -> bool:
        if find_input is None:
            return False
        for x in (getattr(find_input, "input_boxes", None), getattr(find_input, "input_points", None)):
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                return True
        return False

    def _build_prompt_expanded_find_stage(
        self,
        batch_size: int,
        num_chunk_classes: int,
        device: torch.device,
        base_find_input: Optional[FindStage] = None,
    ) -> FindStage:
        if self._has_nonempty_geometric_prompt(base_find_input):
            raise NotImplementedError(
                "Current stage-1 internal chunking only supports semantic-only batches "
                "without non-empty geometric prompts."
            )

        num_pairs = batch_size * num_chunk_classes
        img_ids = torch.arange(batch_size, device=device, dtype=torch.long).repeat_interleave(num_chunk_classes)
        text_ids = torch.arange(num_chunk_classes, device=device, dtype=torch.long).repeat(batch_size)

        return FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=torch.zeros((0, num_pairs, 4), dtype=torch.float32, device=device),
            input_boxes_mask=torch.zeros((num_pairs, 0), dtype=torch.bool, device=device),
            input_boxes_label=torch.zeros((0, num_pairs), dtype=torch.long, device=device),
            input_points=torch.zeros((0, num_pairs, 2), dtype=torch.float32, device=device),
            input_points_mask=torch.zeros((num_pairs, 0), dtype=torch.bool, device=device),
        )

    @staticmethod
    def _reshape_prompt_first_tensor(
        x: Optional[torch.Tensor],
        batch_size: int,
        num_chunk_classes: int,
        key: str,
    ) -> Optional[torch.Tensor]:
        if x is None:
            return None

        expected = batch_size * num_chunk_classes
        if x.shape[0] != expected:
            raise ValueError(
                f"Cannot reshape key={key}: expected first dim = {expected}, got {tuple(x.shape)}"
            )
        return x.reshape(batch_size, num_chunk_classes, *x.shape[1:])

    def _extract_and_reshape_chunk_outputs(
            self,
            raw_outputs: Dict[str, torch.Tensor],
            batch_size: int,
            num_chunk_classes: int,
    ) -> Dict[str, torch.Tensor]:
        out = {}

        for key in (
                OUTPUT_KEYS.semantic_logits,
                OUTPUT_KEYS.class_query,
                OUTPUT_KEYS.extra_token_aux_logits,
        ):
            if key not in raw_outputs or raw_outputs[key] is None:
                continue

            out[key] = self._reshape_prompt_first_tensor(
                raw_outputs[key],
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
                key=key,
            )

        if OUTPUT_KEYS.semantic_logits in out:
            out[OUTPUT_KEYS.semantic_logits] = self._ensure_4d_logits(
                out[OUTPUT_KEYS.semantic_logits],
                OUTPUT_KEYS.semantic_logits,
            )

        return out

    @staticmethod
    def _merge_chunk_outputs(chunk_outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(chunk_outputs) == 0:
            raise ValueError("chunk_outputs is empty.")

        merged = {}
        for key in sorted({k for chunk_out in chunk_outputs for k in chunk_out.keys()}):
            values = [chunk_out[key] for chunk_out in chunk_outputs if key in chunk_out]
            if values:
                merged[key] = torch.cat(values, dim=1)
        return merged

    @staticmethod
    def _ensure_4d_logits(x: torch.Tensor, key: str) -> torch.Tensor:
        if x.dim() == 5:
            if x.shape[2] != 1:
                raise ValueError(
                    f"Expected {key} as [B, C, 1, H, W] when 5D, "
                    f"got {tuple(x.shape)}."
                )
            x = x[:, :, 0]

        if x.dim() != 4:
            raise ValueError(
                f"Expected {key} as [B, C, H, W], got {tuple(x.shape)}."
            )

        return x

    def run_final_mixer(
            self,
            semantic_logits: torch.Tensor,
            clip_dense_logits: torch.Tensor,
            class_query: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._ensure_4d_logits(
            semantic_logits,
            OUTPUT_KEYS.semantic_logits,
        )
        clip_dense_logits = self._ensure_4d_logits(
            clip_dense_logits,
            OUTPUT_KEYS.clip_dense_logits,
        )

        if semantic_logits.shape != clip_dense_logits.shape:
            raise ValueError(
                "semantic_logits and clip_dense_logits must have the same shape, "
                f"got {tuple(semantic_logits.shape)} and {tuple(clip_dense_logits.shape)}."
            )

        if class_query.dim() != 3:
            raise ValueError(
                f"class_query must be [B, C, D], got {tuple(class_query.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape

        if class_query.shape[:2] != (batch_size, num_classes):
            raise ValueError(
                "class_query shape mismatch: "
                f"class_query.shape[:2]={tuple(class_query.shape[:2])}, "
                f"expected={(batch_size, num_classes)}."
            )

        class_query = self.class_query_proj(class_query)

        query_delta, _ = self.class_query_self_attn(
            query=class_query,
            key=class_query,
            value=class_query,
            need_weights=False,
        )
        class_query_context = self.class_query_self_attn_norm(class_query + query_delta)

        fused_memory = self.score_fusion_block(
            semantic_logits=semantic_logits,
            clip_dense_logits=clip_dense_logits,
        )

        memory_tokens = fused_memory.flatten(3).permute(0, 1, 3, 2).contiguous()
        memory_tokens = memory_tokens.reshape(
            batch_size * num_classes,
            height * width,
            self.final_mixer_hidden_dim,
        )

        query_tokens = class_query_context.reshape(
            batch_size * num_classes,
            self.final_mixer_hidden_dim,
        ).unsqueeze(1)

        attended_query, _ = self.suppression_query_cross_attn(
            query=query_tokens,
            key=memory_tokens,
            value=memory_tokens,
            need_weights=False,
        )

        attended_query = self.suppression_query_cross_attn_norm(
            query_tokens + attended_query
        )
        attended_query = attended_query.squeeze(1).reshape(
            batch_size,
            num_classes,
            self.final_mixer_hidden_dim,
        )

        suppression_logits = torch.einsum(
            "bcd,bcdhw->bchw",
            attended_query,
            fused_memory,
        )
        suppression_logits = suppression_logits * self.suppression_logit_scale

        suppression_gate = torch.sigmoid(
            suppression_logits + self.suppression_gate_bias
        )

        pos_logits = semantic_logits.clamp_min(0.0)
        neg_logits = semantic_logits.clamp_max(0.0)

        pos_scale = suppression_gate
        neg_scale = 1.0 + (1.0 - suppression_gate)

        final_logits = pos_logits * pos_scale + neg_logits * neg_scale

        return {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.clip_dense_logits: clip_dense_logits,
            OUTPUT_KEYS.class_query: class_query_context,
            OUTPUT_KEYS.suppression_logits: suppression_logits,
            OUTPUT_KEYS.suppression_gate: suppression_gate,
            OUTPUT_KEYS.final_logits: final_logits,
        }

    def run_final_mixer_from_chunks(
            self,
            mixer_cache: List[Dict[str, torch.Tensor]],
            batch: Optional[BatchedDatapoint] = None,
    ) -> Dict[str, torch.Tensor]:
        if len(mixer_cache) == 0:
            raise ValueError("mixer_cache is empty.")

        mixer_cache = sorted(
            mixer_cache,
            key=lambda item: int(item["chunk_class_ids"][0]),
        )

        semantic_logits = torch.cat(
            [item[OUTPUT_KEYS.semantic_logits] for item in mixer_cache],
            dim=1,
        )
        clip_dense_logits = torch.cat(
            [item[OUTPUT_KEYS.clip_dense_logits] for item in mixer_cache],
            dim=1,
        )
        class_query = torch.cat(
            [item[OUTPUT_KEYS.class_query] for item in mixer_cache],
            dim=1,
        )

        merged_class_ids = []
        for item in mixer_cache:
            merged_class_ids.extend([int(x) for x in item["chunk_class_ids"]])

        expected_class_ids = list(range(len(merged_class_ids)))
        if merged_class_ids != expected_class_ids:
            raise ValueError(
                "mixer_cache chunks must cover classes in order without gaps. "
                f"Got {merged_class_ids}, expected {expected_class_ids}."
            )

        return self.run_final_mixer(
            semantic_logits=semantic_logits,
            clip_dense_logits=clip_dense_logits,
            class_query=class_query,
        )

    def _get_img_feats(self, backbone_out, img_ids):
        vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
        img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
        return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes

    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text=True,
    ):
        txt_feats = backbone_out["language_features"][:, find_input.text_ids]
        txt_masks = backbone_out["language_mask"][find_input.text_ids]

        clip_txt_feats = backbone_out.get("clip_language_features_pair")
        clip_txt_masks = backbone_out.get("clip_language_mask_pair")

        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros((0, *geo_feats.shape[1:]), device=geo_feats.device)
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )

        if not encode_text:
            return (
                torch.cat([geo_feats, visual_prompt_embed], dim=0),
                torch.cat([geo_masks, visual_prompt_mask], dim=1),
                backbone_out,
            )

        prompt_list = [txt_feats]
        prompt_mask_list = [txt_masks]
        if clip_txt_feats is not None:
            prompt_list.append(clip_txt_feats)
            prompt_mask_list.append(clip_txt_masks)

        prompt_list.extend([geo_feats, visual_prompt_embed])
        prompt_mask_list.extend([geo_masks, visual_prompt_mask])
        return torch.cat(prompt_list, dim=0), torch.cat(prompt_mask_list, dim=1), backbone_out

    def _run_encoder(
        self,
        backbone_out,
        find_input,
        prompt,
        prompt_mask,
        encoder_extra_kwargs: Optional[Dict] = None,
    ):
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=torch.zeros_like(prompt),
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

        encoder_out = {
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return backbone_out, encoder_out, feat_tuple

    def _run_semantic_segmentation_head(
        self,
        backbone_out,
        find_input,
        encoder_out,
        prompt,
        prompt_mask,
    ) -> Dict[str, torch.Tensor]:
        if self.segmentation_head is None:
            raise ValueError("segmentation_head is None in semantic mode.")

        seg_outputs = self.segmentation_head(
            backbone_feats=backbone_out["backbone_fpn"],
            obj_queries=torch.empty(0, device=prompt.device),
            image_ids=find_input.img_ids,
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        semantic_logits = seg_outputs.get("semantic_seg")
        if semantic_logits is None:
            raise ValueError("segmentation_head did not return 'semantic_seg' in semantic mode.")
        return {OUTPUT_KEYS.semantic_logits: semantic_logits}

    def forward_grounding_raw(
        self,
        backbone_out: Dict[str, torch.Tensor],
        find_input,
        geometric_prompt: Prompt,
    ) -> Dict[str, torch.Tensor]:
        with torch.profiler.record_function("Sam3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out,
                find_input,
                geometric_prompt,
            )

        with torch.profiler.record_function("Sam3Image._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out,
                find_input,
                prompt,
                prompt_mask,
            )

        with torch.profiler.record_function("Sam3Image._run_semantic_segmentation_head"):
            out = self._run_semantic_segmentation_head(
                backbone_out=backbone_out,
                find_input=find_input,
                encoder_out=encoder_out,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )

        out[OUTPUT_KEYS.extra_token_aux_logits] = self._build_extra_token_aux_logits(
            extra_tokens=backbone_out["extra_tokens_pair"],
            extra_token_mask=backbone_out["extra_tokens_mask_pair"],
            encoder_out=encoder_out,
        )
        out[OUTPUT_KEYS.class_query] = self._run_class_query_encoder_cross_attn(
            class_query_seed=backbone_out["class_query_seed_pair"],
            encoder_out=encoder_out,
        )
        return out

    def forward(self, input: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        chunk_outputs = [chunk["raw_outputs"] for chunk in self.iter_chunk_raw_outputs(input)]
        return self._merge_chunk_outputs(chunk_outputs)
