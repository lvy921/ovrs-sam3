from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.task_modes import OUTPUT_KEYS


@dataclass
class SemanticCriterionConfig:
    ignore_index: int = 255

    bce_weight: float = 1.0
    dice_weight: float = 1.0

    final_bce_weight: float = 0.4
    final_dice_weight: float = 0.5
    final_ce_weight: float = 1.0

    extra_token_aux_loss_weight: float = 0.1
    extra_token_aux_bce_weight: float = 0.03
    extra_token_aux_dice_weight: float = 0.03
    extra_token_aux_absent_weight: float = 0.01
    extra_token_aux_absent_topk_ratio: float = 0.05
    extra_token_aux_exclude_bg: bool = True
    extra_token_aux_bg_idx: int = 0

    suppression_absent_loss_weight: float = 0.05
    suppression_absent_topk_ratio: float = 0.05

    bce_class_balance_clamp_min: float = 0.2
    bce_class_balance_clamp_max: float = 5.0
    eps: float = 1e-6


class SemanticCriterion(nn.Module):
    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()

    @staticmethod
    def _extract_optional_logits(
        outputs: Dict[str, torch.Tensor],
        key: str,
    ) -> Optional[torch.Tensor]:
        logits = outputs.get(key, None)
        if logits is None:
            return None
        if logits.dim() != 4:
            raise ValueError(
                f"Expected {key} as [B, C, H, W], got {tuple(logits.shape)}."
            )
        return logits

    @staticmethod
    def _extract_required_logits(
        outputs: Dict[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        logits = outputs.get(key, None)
        if logits is None:
            raise ValueError(f"SemanticCriterion expects outputs['{key}'].")
        if logits.dim() != 4:
            raise ValueError(
                f"Expected {key} as [B, C, H, W], got {tuple(logits.shape)}."
            )
        return logits

    def _extract_label_map(
        self,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "label_map" not in targets:
            raise ValueError("SemanticCriterion expects targets['label_map'].")

        label_map = targets["label_map"]
        if label_map.dim() == 4:
            if label_map.shape[1] != 1:
                raise ValueError(
                    f"Expected label_map as [B, 1, H, W] or [B, H, W], "
                    f"got {tuple(label_map.shape)}."
                )
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(
                f"Expected label_map as [B, H, W] or [B, 1, H, W], "
                f"got {tuple(label_map.shape)}."
            )

        return label_map.long()

    @staticmethod
    def _resize_label_map_to_logits(
        label_map: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        if tuple(label_map.shape[-2:]) == tuple(target_hw):
            return label_map

        return F.interpolate(
            label_map[:, None].float(),
            size=target_hw,
            mode="nearest",
        )[:, 0].long()

    def _make_zero_losses(
        self,
        ref: torch.Tensor,
        include_extra_aux: bool = True,
        include_final: bool = True,
        include_suppression: bool = True,
    ) -> Dict[str, torch.Tensor]:
        zero = ref.sum() * 0.0

        losses = {
            "loss_semantic_bce": zero,
            "loss_semantic_dice": zero,
        }

        if include_final:
            losses.update(
                {
                    "loss_final_bce": zero,
                    "loss_final_dice": zero,
                    "loss_final_ce": zero,
                }
            )

        if include_extra_aux:
            losses.update(
                {
                    "loss_extra_token_aux_bce": zero,
                    "loss_extra_token_aux_dice": zero,
                    "loss_extra_token_aux_absent": zero,
                    "loss_extra_token_aux": zero,
                }
            )

        if include_suppression:
            losses["loss_suppression_absent"] = zero

        return losses

    def _build_class_ids(
        self,
        num_channels: int,
        chunk_class_ids: Optional[Sequence[int]],
    ) -> list[int]:
        if chunk_class_ids is None:
            return list(range(num_channels))

        if len(chunk_class_ids) != num_channels:
            raise ValueError(
                f"chunk_class_ids length mismatch: expected {num_channels}, "
                f"got {len(chunk_class_ids)}."
            )

        return [int(x) for x in chunk_class_ids]

    def _build_targets(
        self,
        label_map: torch.Tensor,
        class_ids: Sequence[int],
        num_channels: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(class_ids) != num_channels:
            raise ValueError(
                f"class_ids length mismatch: expected {num_channels}, "
                f"got {len(class_ids)}."
            )

        batch_size, height, width = label_map.shape
        valid_mask = label_map != int(self.cfg.ignore_index)

        target = torch.zeros(
            (batch_size, num_channels, height, width),
            dtype=torch.float32,
            device=label_map.device,
        )

        for channel_idx, class_id in enumerate(class_ids):
            target[:, channel_idx] = (label_map == int(class_id)).float()

        valid_mask_4d = valid_mask[:, None].expand(
            batch_size,
            num_channels,
            height,
            width,
        )
        return target, valid_mask_4d

    def _build_present_pair_mask(
        self,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_valid = target * valid_mask.to(dtype=target.dtype)
        fg_pixels_per_pair = target_valid.flatten(2).sum(dim=2)
        return fg_pixels_per_pair > 0

    def _build_absent_pair_mask(
        self,
        present_pair_mask: torch.Tensor,
        class_ids: Sequence[int],
        exclude_bg: bool,
        bg_idx: int,
    ) -> torch.Tensor:
        absent_pair_mask = ~present_pair_mask

        if exclude_bg:
            absent_pair_mask = absent_pair_mask.clone()
            for local_idx, class_id in enumerate(class_ids):
                if int(class_id) == int(bg_idx):
                    absent_pair_mask[:, local_idx] = False

        return absent_pair_mask

    def _build_dynamic_class_weights(
        self,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_valid = target * valid_mask.to(dtype=target.dtype)
        fg_pixels = target_valid.flatten(2).sum(dim=2)

        class_weights = torch.zeros_like(fg_pixels, dtype=target.dtype)

        if present_pair_mask.any():
            present_fg = fg_pixels[present_pair_mask]
            mean_fg = present_fg.mean().clamp_min(1.0)
            class_weights[present_pair_mask] = (
                mean_fg / fg_pixels[present_pair_mask].clamp_min(1.0)
            )

        return class_weights.clamp(
            min=float(self.cfg.bce_class_balance_clamp_min),
            max=float(self.cfg.bce_class_balance_clamp_max),
        )

    def _binary_cross_entropy_present_balanced_mean(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask_4d = present_pair_mask[:, :, None, None]
        effective_mask = valid_mask & pair_mask_4d

        per_elem = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        weight_4d = class_weights[:, :, None, None]
        per_elem = per_elem * weight_4d
        per_elem = per_elem * effective_mask.to(dtype=per_elem.dtype)

        denom = (
            effective_mask.to(dtype=per_elem.dtype) * weight_4d
        ).sum().clamp_min(1.0)
        return per_elem.sum() / denom

    def _dice_loss_present_mean_from_logits(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        prob = logits.sigmoid()
        prob = prob * valid_mask.to(dtype=prob.dtype)
        target = target * valid_mask.to(dtype=target.dtype)

        prob = prob.flatten(2)
        target = target.flatten(2)

        intersection = (prob * target).sum(dim=2)
        denominator = prob.sum(dim=2) + target.sum(dim=2)

        dice = (2.0 * intersection + self.cfg.eps) / (
            denominator + self.cfg.eps
        )
        dice_loss = 1.0 - dice

        pair_weight = present_pair_mask.to(dtype=dice_loss.dtype)
        return (dice_loss * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)

    def _build_ce_label_map(
        self,
        label_map: torch.Tensor,
        class_ids: Sequence[int],
    ) -> torch.Tensor:
        ce_label_map = torch.full_like(
            label_map,
            fill_value=int(self.cfg.ignore_index),
        )

        for local_idx, class_id in enumerate(class_ids):
            ce_label_map[label_map == int(class_id)] = int(local_idx)

        return ce_label_map.long()

    def _cross_entropy_loss(
        self,
        logits: torch.Tensor,
        ce_label_map: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if logits.shape[0] != ce_label_map.shape[0]:
            raise ValueError(
                f"Batch mismatch between logits and ce_label_map: "
                f"{tuple(logits.shape)} vs {tuple(ce_label_map.shape)}."
            )

        if logits.shape[-2:] != ce_label_map.shape[-2:]:
            raise ValueError(
                f"Spatial mismatch between logits and ce_label_map: "
                f"{tuple(logits.shape)} vs {tuple(ce_label_map.shape)}."
            )

        num_valid = int((ce_label_map != int(self.cfg.ignore_index)).sum().item())
        if num_valid <= 0:
            return logits.sum() * 0.0, 0

        loss = F.cross_entropy(
            logits,
            ce_label_map,
            ignore_index=int(self.cfg.ignore_index),
            reduction="mean",
        )
        return loss, num_valid

    def _extra_token_aux_loss(
        self,
        extra_logits: Optional[torch.Tensor],
        label_map: torch.Tensor,
        class_ids: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        ref = extra_logits if extra_logits is not None else label_map.float()
        zero = ref.sum() * 0.0

        if extra_logits is None:
            return {
                "loss_extra_token_aux_bce": zero,
                "loss_extra_token_aux_dice": zero,
                "loss_extra_token_aux_absent": zero,
                "loss_extra_token_aux": zero,
            }

        aux_label_map = self._resize_label_map_to_logits(
            label_map=label_map,
            target_hw=tuple(extra_logits.shape[-2:]),
        )

        target, valid_mask = self._build_targets(
            label_map=aux_label_map,
            class_ids=class_ids,
            num_channels=int(extra_logits.shape[1]),
        )
        present_pair_mask = self._build_present_pair_mask(
            target=target,
            valid_mask=valid_mask,
        )

        if present_pair_mask.any():
            class_weights = self._build_dynamic_class_weights(
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )
            loss_bce = self._binary_cross_entropy_present_balanced_mean(
                logits=extra_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
                class_weights=class_weights,
            )
            loss_dice = self._dice_loss_present_mean_from_logits(
                logits=extra_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )
        else:
            loss_bce = zero
            loss_dice = zero

        absent_pair_mask = self._build_absent_pair_mask(
            present_pair_mask=present_pair_mask,
            class_ids=class_ids,
            exclude_bg=bool(self.cfg.extra_token_aux_exclude_bg),
            bg_idx=int(self.cfg.extra_token_aux_bg_idx),
        )

        if absent_pair_mask.any():
            absent_logits = extra_logits[absent_pair_mask].flatten(1)
            topk_ratio = max(
                0.0,
                min(1.0, float(self.cfg.extra_token_aux_absent_topk_ratio)),
            )
            topk = max(1, int(absent_logits.shape[1] * topk_ratio))
            top_logits = absent_logits.topk(k=topk, dim=1).values
            loss_absent = F.softplus(top_logits).mean()
        else:
            loss_absent = zero

        loss_extra_token_aux = float(self.cfg.extra_token_aux_loss_weight) * (
            float(self.cfg.extra_token_aux_bce_weight) * loss_bce
            + float(self.cfg.extra_token_aux_dice_weight) * loss_dice
            + float(self.cfg.extra_token_aux_absent_weight) * loss_absent
        )

        return {
            "loss_extra_token_aux_bce": loss_bce,
            "loss_extra_token_aux_dice": loss_dice,
            "loss_extra_token_aux_absent": loss_absent,
            "loss_extra_token_aux": loss_extra_token_aux,
        }

    def _suppression_absent_loss(
        self,
        suppression_logits: Optional[torch.Tensor],
        present_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        ref = suppression_logits if suppression_logits is not None else present_pair_mask.float()
        zero = ref.sum() * 0.0

        if suppression_logits is None:
            return zero

        absent_pair_mask = ~present_pair_mask
        if not absent_pair_mask.any():
            return zero

        absent_logits = suppression_logits[absent_pair_mask].flatten(1)

        topk_ratio = max(
            0.0,
            min(1.0, float(self.cfg.suppression_absent_topk_ratio)),
        )
        topk = max(1, int(absent_logits.shape[1] * topk_ratio))

        top_logits = absent_logits.topk(k=topk, dim=1).values
        return F.softplus(top_logits).mean()

    def _forward_chunk(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        chunk_class_ids: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._extract_required_logits(
            outputs,
            OUTPUT_KEYS.semantic_logits,
        )
        extra_token_aux_logits = self._extract_optional_logits(
            outputs,
            OUTPUT_KEYS.extra_token_aux_logits,
        )

        label_map = self._extract_label_map(targets)
        label_map = self._resize_label_map_to_logits(
            label_map=label_map,
            target_hw=tuple(semantic_logits.shape[-2:]),
        )

        num_channels = int(semantic_logits.shape[1])
        class_ids = self._build_class_ids(
            num_channels=num_channels,
            chunk_class_ids=chunk_class_ids,
        )

        target, valid_mask = self._build_targets(
            label_map=label_map,
            class_ids=class_ids,
            num_channels=num_channels,
        )

        present_pair_mask = self._build_present_pair_mask(
            target=target,
            valid_mask=valid_mask,
        )

        extra_aux_losses = self._extra_token_aux_loss(
            extra_logits=extra_token_aux_logits,
            label_map=label_map,
            class_ids=class_ids,
        )

        num_valid_pixels = int((label_map != int(self.cfg.ignore_index)).sum().item())
        zero_losses = self._make_zero_losses(semantic_logits)

        if num_valid_pixels <= 0:
            total_loss = semantic_logits.sum() * 0.0
            return {
                **zero_losses,
                "total_loss": total_loss,
                "num_valid": torch.tensor(
                    0,
                    device=semantic_logits.device,
                    dtype=torch.long,
                ),
            }

        if present_pair_mask.any():
            class_weights = self._build_dynamic_class_weights(
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )

            loss_semantic_bce = self._binary_cross_entropy_present_balanced_mean(
                logits=semantic_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
                class_weights=class_weights,
            )
            loss_semantic_dice = self._dice_loss_present_mean_from_logits(
                logits=semantic_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )
        else:
            loss_semantic_bce = semantic_logits.sum() * 0.0
            loss_semantic_dice = semantic_logits.sum() * 0.0

        total_loss = (
            float(self.cfg.bce_weight) * loss_semantic_bce
            + float(self.cfg.dice_weight) * loss_semantic_dice
            + extra_aux_losses["loss_extra_token_aux"]
        )

        return {
            "loss_semantic_bce": loss_semantic_bce,
            "loss_semantic_dice": loss_semantic_dice,
            "loss_final_bce": zero_losses["loss_final_bce"],
            "loss_final_dice": zero_losses["loss_final_dice"],
            "loss_final_ce": zero_losses["loss_final_ce"],
            "loss_extra_token_aux_bce": extra_aux_losses["loss_extra_token_aux_bce"],
            "loss_extra_token_aux_dice": extra_aux_losses["loss_extra_token_aux_dice"],
            "loss_extra_token_aux_absent": extra_aux_losses[
                "loss_extra_token_aux_absent"
            ],
            "loss_extra_token_aux": extra_aux_losses["loss_extra_token_aux"],
            "loss_suppression_absent": zero_losses["loss_suppression_absent"],
            "total_loss": total_loss,
            "num_valid": torch.tensor(
                num_valid_pixels,
                device=semantic_logits.device,
                dtype=torch.long,
            ),
        }

    def _forward_final(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._extract_required_logits(
            outputs,
            OUTPUT_KEYS.semantic_logits,
        )
        final_logits = self._extract_required_logits(
            outputs,
            OUTPUT_KEYS.final_logits,
        )
        suppression_logits = self._extract_required_logits(
            outputs,
            OUTPUT_KEYS.suppression_logits,
        )

        if final_logits.shape != semantic_logits.shape:
            raise ValueError(
                "final_logits and semantic_logits must have the same shape, "
                f"got {tuple(final_logits.shape)} and {tuple(semantic_logits.shape)}."
            )

        if suppression_logits.shape != semantic_logits.shape:
            raise ValueError(
                "suppression_logits and semantic_logits must have the same shape, "
                f"got {tuple(suppression_logits.shape)} and "
                f"{tuple(semantic_logits.shape)}."
            )

        label_map = self._extract_label_map(targets)
        label_map = self._resize_label_map_to_logits(
            label_map=label_map,
            target_hw=tuple(final_logits.shape[-2:]),
        )

        num_channels = int(final_logits.shape[1])
        class_ids = list(range(num_channels))

        target, valid_mask = self._build_targets(
            label_map=label_map,
            class_ids=class_ids,
            num_channels=num_channels,
        )
        present_pair_mask = self._build_present_pair_mask(
            target=target,
            valid_mask=valid_mask,
        )

        num_valid_pixels = int((label_map != int(self.cfg.ignore_index)).sum().item())
        zero_losses = self._make_zero_losses(final_logits)

        if num_valid_pixels <= 0:
            total_loss = final_logits.sum() * 0.0
            return {
                **zero_losses,
                "total_loss": total_loss,
                "num_valid": torch.tensor(
                    0,
                    device=final_logits.device,
                    dtype=torch.long,
                ),
            }

        if present_pair_mask.any():
            class_weights = self._build_dynamic_class_weights(
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )
            loss_final_bce = self._binary_cross_entropy_present_balanced_mean(
                logits=final_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
                class_weights=class_weights,
            )
            loss_final_dice = self._dice_loss_present_mean_from_logits(
                logits=final_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )
        else:
            loss_final_bce = final_logits.sum() * 0.0
            loss_final_dice = final_logits.sum() * 0.0

        ce_label_map = self._build_ce_label_map(
            label_map=label_map,
            class_ids=class_ids,
        )
        loss_final_ce, num_ce_valid = self._cross_entropy_loss(
            logits=final_logits,
            ce_label_map=ce_label_map,
        )

        raw_loss_suppression_absent = self._suppression_absent_loss(
            suppression_logits=suppression_logits,
            present_pair_mask=present_pair_mask,
        )
        loss_suppression_absent = (
            float(self.cfg.suppression_absent_loss_weight)
            * raw_loss_suppression_absent
        )

        total_loss = (
            float(self.cfg.final_bce_weight) * loss_final_bce
            + float(self.cfg.final_dice_weight) * loss_final_dice
            + float(self.cfg.final_ce_weight) * loss_final_ce
            + loss_suppression_absent
        )

        return {
            "loss_semantic_bce": zero_losses["loss_semantic_bce"],
            "loss_semantic_dice": zero_losses["loss_semantic_dice"],
            "loss_final_bce": loss_final_bce,
            "loss_final_dice": loss_final_dice,
            "loss_final_ce": loss_final_ce,
            "loss_extra_token_aux_bce": zero_losses["loss_extra_token_aux_bce"],
            "loss_extra_token_aux_dice": zero_losses["loss_extra_token_aux_dice"],
            "loss_extra_token_aux_absent": zero_losses[
                "loss_extra_token_aux_absent"
            ],
            "loss_extra_token_aux": zero_losses["loss_extra_token_aux"],
            "loss_suppression_absent": raw_loss_suppression_absent,
            "total_loss": total_loss,
            "num_valid": torch.tensor(
                max(num_valid_pixels, num_ce_valid),
                device=final_logits.device,
                dtype=torch.long,
            ),
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        chunk_class_ids: Optional[Sequence[int]] = None,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        if reduction != "mean":
            raise ValueError(
                f"SemanticCriterion only supports reduction='mean', got {reduction!r}."
            )

        is_final_stage = (
            OUTPUT_KEYS.final_logits in outputs
            and OUTPUT_KEYS.suppression_logits in outputs
        )

        if is_final_stage:
            return self._forward_final(
                outputs=outputs,
                targets=targets,
            )

        if chunk_class_ids is None:
            raise ValueError(
                "Chunk-stage SemanticCriterion requires chunk_class_ids."
            )

        return self._forward_chunk(
            outputs=outputs,
            targets=targets,
            chunk_class_ids=chunk_class_ids,
        )


class HybridCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HybridCriterion is not implemented yet.")