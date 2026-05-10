from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..models.task_modes import OUTPUT_KEYS


@dataclass
class VisualizerConfig:
    enabled: bool = False
    save_dir: str = "./visualizations"
    save_stage: str = "val"
    alpha: float = 0.45

    save_original: bool = True
    save_prediction: bool = True
    save_ground_truth: bool = True
    save_semantic_prediction: bool = True

    save_score_summary: bool = True
    save_score_heatmaps: bool = True
    save_extra_token_aux_heatmaps: bool = True
    heatmap_colormap: str = "turbo"

    save_clip_argmax_prediction: bool = True

    vis_prob: float = 0.05
    max_samples_per_epoch: Optional[int] = 50
    vis_seed: int = 42

    image_folder_pattern: str = "image_{image_id:06d}"
    ignore_index: int = 255


@dataclass
class VisualizationContext:
    model: torch.nn.Module
    batch: Any
    semantic_outputs: Dict[str, torch.Tensor]
    semantic_targets: Dict[str, torch.Tensor]
    epoch: Optional[int]
    stage: str
    selected_indices: List[int]


class VisualizationTask:
    name = "base"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        raise NotImplementedError


class BaseSemanticOverlayTask(VisualizationTask):
    name = "base_semantic_overlay"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        semantic_outputs = ctx.semantic_outputs
        semantic_targets = ctx.semantic_targets
        batch = ctx.batch

        if OUTPUT_KEYS.final_score_map not in semantic_outputs:
            raise ValueError(f"outputs must contain '{OUTPUT_KEYS.final_score_map}'.")

        final_score_map = semantic_outputs[OUTPUT_KEYS.final_score_map]
        final_pred = manager._extract_pred_from_logits(final_score_map)

        semantic_score_map = semantic_outputs.get(OUTPUT_KEYS.semantic_score_map, None)
        semantic_pred = (
            manager._extract_pred_from_logits(semantic_score_map)
            if semantic_score_map is not None
            else None
        )

        gt = semantic_targets["label_map"]
        if gt.dim() == 4:
            if gt.shape[1] != 1:
                raise ValueError(
                    f"Expected gt as [B,1,H,W] or [B,H,W], got {tuple(gt.shape)}"
                )
            gt = gt[:, 0]

        pred_num_classes = int(final_score_map.shape[1])

        try:
            class_names: List[str] = batch.find_metadatas[0].class_names
            gt_num_classes = len(class_names)
        except Exception:
            gt_num_classes = manager._infer_num_classes_from_label_map(
                gt,
                ignore_index=manager.cfg.ignore_index,
                fallback=pred_num_classes,
            )
            class_names = None

        semantic_num_classes = pred_num_classes
        if semantic_score_map is not None:
            semantic_num_classes = int(semantic_score_map.shape[1])

        for b in ctx.selected_indices:
            image_id = manager._extract_image_id(batch, b)
            sample_dir = manager._resolve_sample_dir(
                image_id=image_id,
                epoch=ctx.epoch,
                stage=ctx.stage,
            )

            overlay_image = manager._extract_overlay_image(batch, b)
            original_image = manager._extract_original_reference_image(batch, b)
            out_hw = overlay_image.size[::-1]

            final_pred_label = manager._prepare_label_map(final_pred[b], out_hw)
            gt_label = manager._prepare_label_map(gt[b], out_hw)

            if manager.cfg.save_original:
                original_image.save(sample_dir / "original.png")

            if manager.cfg.save_prediction:
                manager._overlay_label_map(
                    overlay_image,
                    final_pred_label,
                    pred_num_classes,
                ).save(sample_dir / "pred_overlay.png")

            if manager.cfg.save_ground_truth:
                manager._overlay_label_map(
                    overlay_image,
                    gt_label,
                    gt_num_classes,
                ).save(sample_dir / "gt_overlay.png")

            if semantic_pred is not None and manager.cfg.save_semantic_prediction:
                semantic_pred_label = manager._prepare_label_map(semantic_pred[b], out_hw)
                manager._overlay_label_map(
                    overlay_image,
                    semantic_pred_label,
                    semantic_num_classes,
                ).save(sample_dir / "pred_semantic_overlay.png")

            if class_names is not None:
                with open(sample_dir / "classes.txt", "w", encoding="utf-8") as f:
                    for i, name in enumerate(class_names):
                        f.write(f"{i}\t{name}\n")


class BranchScoreAnalysisTask(VisualizationTask):
    name = "branch_score_analysis"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        outputs = ctx.semantic_outputs
        batch = ctx.batch

        semantic_score_map = outputs.get(OUTPUT_KEYS.semantic_score_map, None)
        final_score_map = outputs.get(OUTPUT_KEYS.final_score_map, None)

        if semantic_score_map is None:
            semantic_logits = outputs.get(OUTPUT_KEYS.semantic_logits, None)
            if semantic_logits is not None:
                semantic_score_map = semantic_logits.sigmoid()

        if final_score_map is None:
            final_logits = outputs.get(OUTPUT_KEYS.final_logits, None)
            if final_logits is not None:
                final_score_map = final_logits.sigmoid()

        extra_token_aux_logits = outputs.get(OUTPUT_KEYS.extra_token_aux_logits, None)

        if semantic_score_map is None:
            raise ValueError(
                f"outputs must contain '{OUTPUT_KEYS.semantic_score_map}' "
                f"or '{OUTPUT_KEYS.semantic_logits}'."
            )
        if final_score_map is None:
            raise ValueError(
                f"outputs must contain '{OUTPUT_KEYS.final_score_map}' "
                f"or '{OUTPUT_KEYS.final_logits}'."
            )

        if semantic_score_map.dim() != 4:
            raise ValueError(
                f"Expected semantic_score_map as [B, C, H, W], "
                f"got {tuple(semantic_score_map.shape)}."
            )
        if final_score_map.dim() != 4:
            raise ValueError(
                f"Expected final_score_map as [B, C, H, W], "
                f"got {tuple(final_score_map.shape)}."
            )
        if semantic_score_map.shape != final_score_map.shape:
            raise ValueError(
                "semantic_score_map and final_score_map shape mismatch: "
                f"{tuple(semantic_score_map.shape)} vs {tuple(final_score_map.shape)}."
            )

        if extra_token_aux_logits is not None:
            if extra_token_aux_logits.dim() != 4:
                raise ValueError(
                    "Expected extra_token_aux_logits as [B, C, H, W], "
                    f"got {tuple(extra_token_aux_logits.shape)}."
                )
            if extra_token_aux_logits.shape[:2] != semantic_score_map.shape[:2]:
                raise ValueError(
                    "extra_token_aux_logits batch/class shape mismatch: "
                    f"{tuple(extra_token_aux_logits.shape[:2])} vs "
                    f"{tuple(semantic_score_map.shape[:2])}."
                )

        try:
            class_names = [str(x) for x in batch.find_metadatas[0].class_names]
        except Exception:
            class_names = None

        for b in ctx.selected_indices:
            image_id = manager._extract_image_id(batch, b)
            sample_dir = manager._resolve_sample_dir(
                image_id=image_id,
                epoch=ctx.epoch,
                stage=ctx.stage,
            )

            overlay_image = manager._extract_overlay_image(batch, b)
            out_hw = overlay_image.size[::-1]

            semantic_scores_b = semantic_score_map[b]
            final_scores_b = final_score_map[b]
            extra_token_aux_logits_b = (
                extra_token_aux_logits[b]
                if extra_token_aux_logits is not None
                else None
            )

            if manager.cfg.save_score_summary:
                manager._save_score_summary(
                    sample_dir=sample_dir,
                    semantic_scores=semantic_scores_b,
                    final_scores=final_scores_b,
                    extra_token_aux_logits=extra_token_aux_logits_b,
                    class_names=class_names,
                )

            if manager.cfg.save_score_heatmaps:
                manager._save_all_score_heatmaps(
                    sample_dir=sample_dir,
                    semantic_scores=semantic_scores_b,
                    final_scores=final_scores_b,
                    extra_token_aux_logits=extra_token_aux_logits_b,
                    out_hw=out_hw,
                    class_names=class_names,
                )


class ClipImageTextScoreTask(VisualizationTask):
    name = "clip_image_text_score"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        if not manager.cfg.save_clip_argmax_prediction:
            return

        clip_score_map = manager._build_clip_image_text_score_map_for_visualization(
            model=ctx.model,
            batch=ctx.batch,
        )
        if clip_score_map is None:
            return

        if clip_score_map.dim() != 4:
            raise ValueError(
                f"Expected clip_score_map as [B, C, Hc, Wc], "
                f"got {tuple(clip_score_map.shape)}."
            )

        num_classes = int(clip_score_map.shape[1])

        for b in ctx.selected_indices:
            image_id = manager._extract_image_id(ctx.batch, b)
            sample_dir = manager._resolve_sample_dir(
                image_id=image_id,
                epoch=ctx.epoch,
                stage=ctx.stage,
            )

            overlay_image = manager._extract_overlay_image(ctx.batch, b)
            out_hw = overlay_image.size[::-1]

            clip_score_up = F.interpolate(
                clip_score_map[b:b + 1],
                size=out_hw,
                mode="bilinear",
                align_corners=False,
            )[0]

            clip_pred = clip_score_up.argmax(dim=0).long()

            manager._colorize_label_map(
                clip_pred.detach().cpu(),
                num_classes=num_classes,
            ).save(sample_dir / "clip_argmax_pred.png")

            manager._overlay_label_map(
                overlay_image,
                clip_pred.detach().cpu(),
                num_classes=num_classes,
            ).save(sample_dir / "clip_argmax_overlay.png")


class VisualizationManager:
    def __init__(self, cfg: VisualizerConfig):
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._saved_counts: Dict[Tuple[str, int], int] = {}
        self.tasks = self._build_tasks()

    def _build_tasks(self):
        return [
            BaseSemanticOverlayTask(),
            BranchScoreAnalysisTask(),
            ClipImageTextScoreTask(),
        ]

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return getattr(model, "module", model)

    @classmethod
    def _extract_core_model(cls, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        model = cls._unwrap_model(model)
        return getattr(model, "core", None)

    def _build_clip_image_text_score_map_for_visualization(
            self,
            model: torch.nn.Module,
            batch: Any,
    ) -> Optional[torch.Tensor]:
        core = self._extract_core_model(model)
        if core is None:
            return None

        required_attrs = [
            "clip_image_encoder",
            "clip_text_encoder",
            "_prepare_openclip_image_batch",
        ]
        for name in required_attrs:
            if not hasattr(core, name):
                return None

        if core.clip_image_encoder is None or core.clip_text_encoder is None:
            return None

        class_texts = list(getattr(batch, "find_text_batch", []))
        if len(class_texts) == 0:
            return None

        raw_images = getattr(batch, "raw_images", None)
        if raw_images is None:
            return None

        templates = list(getattr(core, "clip_extra_token_templates", []))
        if len(templates) == 0:
            return None

        normalize_label = bool(getattr(core, "normalize_label_for_clip", True))
        device = core.device

        with torch.no_grad():
            clip_img_batch = core._prepare_openclip_image_batch(
                raw_images=raw_images,
                device=device,
            )

            clip_image_feat_map = core.clip_image_encoder(clip_img_batch)
            if not isinstance(clip_image_feat_map, torch.Tensor):
                return None
            if clip_image_feat_map.dim() != 4:
                raise ValueError(
                    "clip_image_encoder must return [B, D_clip, Hc, Wc], "
                    f"got {tuple(clip_image_feat_map.shape)}"
                )

            clip_text_tokens = core.clip_text_encoder.encode_prompt_templates(
                class_names=class_texts,
                templates=templates,
                device=device,
                normalize_label=normalize_label,
                normalize=False,
            )

            if clip_text_tokens.dim() != 3:
                raise ValueError(
                    "Expected clip_text_tokens as [C, K, D_clip], "
                    f"got {tuple(clip_text_tokens.shape)}"
                )

            batch_size, image_dim, grid_h, grid_w = clip_image_feat_map.shape
            num_classes, _, text_dim = clip_text_tokens.shape

            if image_dim != text_dim:
                raise ValueError(
                    f"CLIP image/text dim mismatch: image_dim={image_dim}, "
                    f"text_dim={text_dim}"
                )

            if num_classes != len(class_texts):
                raise ValueError(
                    f"CLIP text class count mismatch: {num_classes} vs {len(class_texts)}"
                )

            image_features = clip_image_feat_map.flatten(2).transpose(1, 2).contiguous()
            image_features = F.normalize(image_features, dim=-1, eps=1e-6)

            text_features = F.normalize(clip_text_tokens, dim=-1, eps=1e-6)
            text_features = text_features.mean(dim=1)
            text_features = F.normalize(text_features, dim=-1, eps=1e-6)

            clip_score = torch.einsum(
                "bnd,cd->bcn",
                image_features,
                text_features,
            )

            clip_score_map = clip_score.reshape(
                batch_size,
                num_classes,
                grid_h,
                grid_w,
            ).contiguous()

        return clip_score_map.detach()

    @classmethod
    def from_cfg(
        cls,
        cfg_dict: Optional[Dict[str, Any]],
        work_dir: Optional[str] = None,
    ) -> Optional["VisualizationManager"]:
        if cfg_dict is None:
            return None
        cfg = VisualizerConfig(**cfg_dict)
        if not cfg.enabled:
            return None

        save_dir = Path(cfg.save_dir)
        if not save_dir.is_absolute() and work_dir is not None:
            cfg.save_dir = str(Path(work_dir) / save_dir)
        return cls(cfg)

    def should_save(self, stage: str) -> bool:
        if not self.cfg.enabled:
            return False
        if self.cfg.save_stage == "all":
            return True
        return self.cfg.save_stage == stage

    @staticmethod
    def _to_uint8_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            x = image.detach().cpu()
            if x.dim() == 4:
                x = x[0]
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            x = x.float().clamp(0, 1)
            arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")

        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")

        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _extract_overlay_image(batch: Any, batch_index: int) -> Image.Image:
        raw_images = getattr(batch, "raw_images", None)
        if (
            raw_images is not None
            and batch_index < len(raw_images)
            and raw_images[batch_index] is not None
        ):
            return VisualizationManager._to_uint8_image(raw_images[batch_index])
        return VisualizationManager._to_uint8_image(batch.img_batch[batch_index])

    @staticmethod
    def _extract_original_reference_image(batch: Any, batch_index: int) -> Image.Image:
        raw_images_original = getattr(batch, "raw_images_original", None)
        if (
            raw_images_original is not None
            and batch_index < len(raw_images_original)
            and raw_images_original[batch_index] is not None
        ):
            return VisualizationManager._to_uint8_image(raw_images_original[batch_index])

        raw_images = getattr(batch, "raw_images", None)
        if (
            raw_images is not None
            and batch_index < len(raw_images)
            and raw_images[batch_index] is not None
        ):
            return VisualizationManager._to_uint8_image(raw_images[batch_index])

        return VisualizationManager._to_uint8_image(batch.img_batch[batch_index])

    @staticmethod
    def _extract_image_id(batch: Any, batch_index: int) -> int:
        try:
            meta = batch.find_metadatas[0]
            return int(meta.original_image_id[batch_index].item())
        except Exception:
            return int(batch_index)

    def _resolve_sample_dir(
        self,
        image_id: int,
        epoch: Optional[int],
        stage: str,
    ) -> Path:
        parts = [self.save_dir, stage]
        if epoch is not None:
            parts.append(Path(f"epoch_{epoch:03d}"))
        parts.append(Path(self.cfg.image_folder_pattern.format(image_id=image_id)))
        sample_dir = Path(*parts)
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir

    @staticmethod
    def _prepare_label_map(
        label_map: torch.Tensor,
        out_hw: Tuple[int, int],
    ) -> torch.Tensor:
        x = label_map.detach().cpu()
        if x.dim() == 3:
            if x.shape[0] != 1:
                raise ValueError(f"Expected [1,H,W] or [H,W], got {tuple(x.shape)}")
            x = x[0]
        if x.dim() != 2:
            raise ValueError(f"Expected [H,W], got {tuple(x.shape)}")

        if tuple(x.shape[-2:]) != tuple(out_hw):
            x = F.interpolate(x[None, None].float(), size=out_hw, mode="nearest")[0, 0].long()
        else:
            x = x.long()
        return x

    @staticmethod
    def _infer_num_classes_from_label_map(
        label_map: torch.Tensor,
        ignore_index: int,
        fallback: int,
    ) -> int:
        x = label_map.detach().cpu().long()
        if x.dim() == 4:
            if x.shape[1] != 1:
                raise ValueError(f"Expected [B,1,H,W] or [B,H,W], got {tuple(x.shape)}")
            x = x[:, 0]
        elif x.dim() != 3:
            raise ValueError(f"Expected [B,H,W], got {tuple(x.shape)}")

        valid = x != int(ignore_index)
        if not valid.any():
            return int(fallback)

        max_label = int(x[valid].max().item())
        return max(int(fallback), max_label + 1)

    @staticmethod
    def _build_palette(num_classes: int) -> np.ndarray:
        palette = np.zeros((num_classes, 3), dtype=np.uint8)
        for i in range(num_classes):
            lab = i
            r = g = b = 0
            for j in range(8):
                r |= ((lab >> 0) & 1) << (7 - j)
                g |= ((lab >> 1) & 1) << (7 - j)
                b |= ((lab >> 2) & 1) << (7 - j)
                lab >>= 3
            palette[i] = np.array([r, g, b], dtype=np.uint8)
        return palette

    def _colorize_label_map(self, label_map: torch.Tensor, num_classes: int) -> Image.Image:
        label_map_np = label_map.cpu().numpy().astype(np.int64)
        h, w = label_map_np.shape

        valid = label_map_np != self.cfg.ignore_index
        if valid.any():
            max_label = int(label_map_np[valid].max())
            num_classes = max(int(num_classes), max_label + 1)
        else:
            num_classes = int(num_classes)

        palette = self._build_palette(num_classes)

        color = np.zeros((h, w, 3), dtype=np.uint8)
        safe_label = label_map_np.copy()
        safe_label[~valid] = 0
        color[valid] = palette[safe_label[valid]]
        return Image.fromarray(color, mode="RGB")

    def _overlay_label_map(
        self,
        image: Image.Image,
        label_map: torch.Tensor,
        num_classes: int,
    ) -> Image.Image:
        base = np.asarray(image.convert("RGB")).astype(np.float32)
        color = np.asarray(self._colorize_label_map(label_map, num_classes)).astype(np.float32)

        valid = (label_map.cpu().numpy() != self.cfg.ignore_index)[..., None]
        out = base.copy()
        out[valid[..., 0]] = (
            (1.0 - self.cfg.alpha) * base[valid[..., 0]]
            + self.cfg.alpha * color[valid[..., 0]]
        )
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    @staticmethod
    def _extract_pred_from_logits(
        logits: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if logits is None:
            return None
        if logits.dim() != 4:
            raise ValueError(f"Expected logits [B,C,H,W], got {tuple(logits.shape)}")
        return logits.argmax(dim=1)

    @staticmethod
    def _apply_turbo_colormap(x: np.ndarray) -> np.ndarray:
        x = np.clip(x.astype(np.float32), 0.0, 1.0)

        coeffs = np.array(
            [
                [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
                [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
                [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
            ],
            dtype=np.float32,
        )

        powers = np.stack(
            [
                np.ones_like(x),
                x,
                x ** 2,
                x ** 3,
                x ** 4,
                x ** 5,
            ],
            axis=-1,
        )

        rgb = powers @ coeffs.T
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)

    @staticmethod
    def _apply_gray_colormap(x: np.ndarray) -> np.ndarray:
        x = np.clip(x.astype(np.float32), 0.0, 1.0)
        arr = (x * 255.0).astype(np.uint8)
        return np.stack([arr, arr, arr], axis=-1)

    def _apply_colormap(self, x: np.ndarray) -> np.ndarray:
        name = str(getattr(self.cfg, "heatmap_colormap", "turbo")).lower()

        if name == "turbo":
            return self._apply_turbo_colormap(x)

        if name in {"gray", "grey", "grayscale"}:
            return self._apply_gray_colormap(x)

        raise ValueError(
            f"Unsupported heatmap_colormap={name!r}. "
            "Supported values are: 'turbo', 'gray'."
        )

    @staticmethod
    def _normalize_heatmap_values(
            x: torch.Tensor,
            normalize: str,
    ) -> torch.Tensor:
        normalize = str(normalize)

        if normalize == "prob":
            return x.clamp(0.0, 1.0)

        if normalize == "sigmoid":
            return x.sigmoid()

        if normalize == "minmax":
            x_min = x.min()
            x_max = x.max()
            return (x - x_min) / (x_max - x_min).clamp_min(1e-6)

        if normalize == "auto":
            x_min = x.min()
            x_max = x.max()

            if float(x_min.item()) >= 0.0 and float(x_max.item()) <= 1.0:
                return x.clamp(0.0, 1.0)

            return (x - x_min) / (x_max - x_min).clamp_min(1e-6)

        raise ValueError(
            f"Unknown heatmap normalize mode={normalize!r}. "
            "Supported modes are: 'auto', 'prob', 'sigmoid', 'minmax'."
        )

    def _to_heatmap_image(
            self,
            score_map: torch.Tensor,
            out_hw: Tuple[int, int],
            normalize: str = "auto",
    ) -> Image.Image:
        x = score_map.detach().cpu().float()

        if x.dim() == 3:
            if x.shape[0] != 1:
                raise ValueError(f"Expected [1, H, W] or [H, W], got {tuple(x.shape)}.")
            x = x[0]

        if x.dim() != 2:
            raise ValueError(f"Expected [H, W], got {tuple(x.shape)}.")

        if tuple(x.shape[-2:]) != tuple(out_hw):
            x = F.interpolate(
                x[None, None],
                size=out_hw,
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        x = self._normalize_heatmap_values(x, normalize=normalize)

        arr = x.numpy()
        heat = self._apply_colormap(arr)
        return Image.fromarray(heat, mode="RGB")

    def _to_normalized_heatmap_image(
            self,
            score_map: torch.Tensor,
            out_hw: Tuple[int, int],
    ) -> Image.Image:
        return self._to_heatmap_image(
            score_map=score_map,
            out_hw=out_hw,
            normalize="minmax",
        )

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        safe = []
        for ch in str(text):
            if ch.isalnum() or ch in ("-", "_"):
                safe.append(ch)
            elif ch in (" ", "/", "\\", "."):
                safe.append("_")
        value = "".join(safe).strip("_")
        return value or "class"

    @staticmethod
    def _format_optional_float(value: Optional[float]) -> str:
        if value is None:
            return "nan"
        return f"{float(value):.6f}"

    @staticmethod
    def _per_class_max_mean(
            score_map: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if score_map is None:
            return None, None

        if score_map.dim() != 3:
            raise ValueError(
                f"Expected score_map as [C, H, W], got {tuple(score_map.shape)}."
            )

        flat = score_map.flatten(1)
        return flat.max(dim=1).values, flat.mean(dim=1)

    def _save_score_summary(
            self,
            sample_dir: Path,
            semantic_scores: torch.Tensor,
            final_scores: torch.Tensor,
            extra_token_aux_logits: Optional[torch.Tensor],
            class_names: Optional[List[str]],
    ) -> None:
        if semantic_scores.dim() != 3:
            raise ValueError(
                f"Expected semantic_scores as [C, H, W], got {tuple(semantic_scores.shape)}."
            )
        if final_scores.dim() != 3:
            raise ValueError(
                f"Expected final_scores as [C, H, W], got {tuple(final_scores.shape)}."
            )

        num_classes = int(semantic_scores.shape[0])
        if final_scores.shape[0] != num_classes:
            raise ValueError(
                f"Class count mismatch: semantic={num_classes}, final={final_scores.shape[0]}."
            )

        semantic_max, semantic_mean = self._per_class_max_mean(semantic_scores)
        final_max, final_mean = self._per_class_max_mean(final_scores)
        extra_aux_max, extra_aux_mean = self._per_class_max_mean(extra_token_aux_logits)

        order = torch.argsort(final_max, descending=True)

        with open(sample_dir / "branch_score_summary.txt", "w", encoding="utf-8") as f:
            f.write(
                "rank\tclass_id\tclass_name\t"
                "semantic_max\tsemantic_mean\t"
                "final_max\tfinal_mean\t"
                "extra_token_aux_logit_max\textra_token_aux_logit_mean\n"
            )

            for rank, cls_idx in enumerate(order.tolist()):
                class_name = (
                    class_names[cls_idx]
                    if class_names is not None and cls_idx < len(class_names)
                    else f"class_{cls_idx}"
                )

                extra_aux_max_value = (
                    float(extra_aux_max[cls_idx].item())
                    if extra_aux_max is not None
                    else None
                )
                extra_aux_mean_value = (
                    float(extra_aux_mean[cls_idx].item())
                    if extra_aux_mean is not None
                    else None
                )

                f.write(
                    f"{rank}\t{cls_idx}\t{class_name}\t"
                    f"{float(semantic_max[cls_idx].item()):.6f}\t"
                    f"{float(semantic_mean[cls_idx].item()):.6f}\t"
                    f"{float(final_max[cls_idx].item()):.6f}\t"
                    f"{float(final_mean[cls_idx].item()):.6f}\t"
                    f"{self._format_optional_float(extra_aux_max_value)}\t"
                    f"{self._format_optional_float(extra_aux_mean_value)}\n"
                )

    def _save_all_score_heatmaps(
            self,
            sample_dir: Path,
            semantic_scores: torch.Tensor,
            final_scores: torch.Tensor,
            extra_token_aux_logits: Optional[torch.Tensor],
            out_hw: Tuple[int, int],
            class_names: Optional[List[str]],
    ) -> None:
        if semantic_scores.dim() != 3:
            raise ValueError(
                f"Expected semantic_scores as [C, H, W], got {tuple(semantic_scores.shape)}."
            )
        if final_scores.dim() != 3:
            raise ValueError(
                f"Expected final_scores as [C, H, W], got {tuple(final_scores.shape)}."
            )

        num_classes = int(semantic_scores.shape[0])
        if final_scores.shape[0] != num_classes:
            raise ValueError(
                f"Class count mismatch in score heatmaps: "
                f"semantic={num_classes}, final={final_scores.shape[0]}."
            )

        if extra_token_aux_logits is not None:
            if extra_token_aux_logits.dim() != 3:
                raise ValueError(
                    "Expected extra_token_aux_logits as [C, H, W], "
                    f"got {tuple(extra_token_aux_logits.shape)}."
                )
            if extra_token_aux_logits.shape[0] != num_classes:
                raise ValueError(
                    "extra_token_aux_logits class count mismatch: "
                    f"{extra_token_aux_logits.shape[0]} vs {num_classes}."
                )

        final_max = final_scores.flatten(1).max(dim=1).values
        order = torch.argsort(final_max, descending=True)

        heatmap_dir = sample_dir / "score_heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        with open(heatmap_dir / "all_classes.txt", "w", encoding="utf-8") as f:
            f.write(
                "rank\tclass_id\tclass_name\t"
                "semantic_max\tfinal_max\t"
                "has_extra_token_aux_logits\n"
            )

            for rank, cls_idx in enumerate(order.tolist()):
                class_name = (
                    class_names[cls_idx]
                    if class_names is not None and cls_idx < len(class_names)
                    else f"class_{cls_idx}"
                )
                safe_name = self._sanitize_filename(class_name)

                semantic_max_value = float(
                    semantic_scores[cls_idx].flatten().max().item()
                )
                final_max_value = float(
                    final_scores[cls_idx].flatten().max().item()
                )

                f.write(
                    f"{rank}\t{cls_idx}\t{class_name}\t"
                    f"{semantic_max_value:.6f}\t"
                    f"{final_max_value:.6f}\t"
                    f"{int(extra_token_aux_logits is not None)}\n"
                )

                semantic_img = self._to_heatmap_image(
                    semantic_scores[cls_idx],
                    out_hw=out_hw,
                    normalize="prob",
                )
                final_img = self._to_heatmap_image(
                    final_scores[cls_idx],
                    out_hw=out_hw,
                    normalize="prob",
                )

                semantic_img.save(
                    heatmap_dir
                    / f"rank_{rank:03d}_class_{cls_idx:03d}_{safe_name}_semantic_score.png"
                )
                final_img.save(
                    heatmap_dir
                    / f"rank_{rank:03d}_class_{cls_idx:03d}_{safe_name}_final_score.png"
                )

                if (
                        self.cfg.save_extra_token_aux_heatmaps
                        and extra_token_aux_logits is not None
                ):
                    extra_aux_img = self._to_heatmap_image(
                        extra_token_aux_logits[cls_idx],
                        out_hw=out_hw,
                        normalize="minmax",
                    )
                    extra_aux_img.save(
                        heatmap_dir
                        / f"rank_{rank:03d}_class_{cls_idx:03d}_{safe_name}_extra_token_aux_logits.png"
                    )

    def _get_epoch_key(self, stage: str, epoch: Optional[int]) -> Tuple[str, int]:
        return stage, (-1 if epoch is None else int(epoch))

    def _get_saved_count(self, stage: str, epoch: Optional[int]) -> int:
        key = self._get_epoch_key(stage, epoch)
        return int(self._saved_counts.get(key, 0))

    def _increase_saved_count(self, stage: str, epoch: Optional[int]) -> None:
        key = self._get_epoch_key(stage, epoch)
        self._saved_counts[key] = self._get_saved_count(stage, epoch) + 1

    def _should_save_sample(
        self,
        image_id: int,
        stage: str,
        epoch: Optional[int],
    ) -> bool:
        if not self.should_save(stage):
            return False

        if self.cfg.vis_prob <= 0:
            return False

        saved_count = self._get_saved_count(stage, epoch)
        if self.cfg.max_samples_per_epoch is not None:
            if saved_count >= int(self.cfg.max_samples_per_epoch):
                return False

        if self.cfg.vis_prob >= 1.0:
            return True

        epoch_value = -1 if epoch is None else int(epoch)
        token = f"{self.cfg.vis_seed}:{stage}:{epoch_value}:{int(image_id)}"
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / float(16 ** 8 - 1)

        return value < float(self.cfg.vis_prob)

    def run(
        self,
        model: torch.nn.Module,
        batch: Any,
        semantic_outputs: Dict[str, torch.Tensor],
        semantic_targets: Dict[str, torch.Tensor],
        *,
        epoch: Optional[int],
        stage: str = "val",
    ) -> None:
        if not self.should_save(stage):
            return

        bsz = int(batch.img_batch.shape[0])
        selected_indices = []

        for b in range(bsz):
            image_id = self._extract_image_id(batch, b)
            if self._should_save_sample(image_id=image_id, stage=stage, epoch=epoch):
                selected_indices.append(b)
                self._increase_saved_count(stage, epoch)

        if len(selected_indices) == 0:
            return

        ctx = VisualizationContext(
            model=model,
            batch=batch,
            semantic_outputs=semantic_outputs,
            semantic_targets=semantic_targets,
            epoch=epoch,
            stage=stage,
            selected_indices=selected_indices,
        )

        for task in self.tasks:
            task.run(self, ctx)