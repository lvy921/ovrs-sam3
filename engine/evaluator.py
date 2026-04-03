from __future__ import annotations

import copy
import math
from dataclasses import fields, is_dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .visualization import VisualizationManager

TensorDict = Dict[str, torch.Tensor]


def move_batch_to_device(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if is_dataclass(obj):
        for field in fields(obj):
            setattr(obj, field.name, move_batch_to_device(getattr(obj, field.name), device))
        return obj
    if isinstance(obj, dict):
        return {k: move_batch_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_batch_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_batch_to_device(v, device) for v in obj)
    return obj


class MulticlassSemanticEvaluator:
    def __init__(self, ignore_index: int = 255):
        self.ignore_index = int(ignore_index)
        self.num_classes: Optional[int] = None
        self.reset()

    def reset(self):
        self.intersection = None
        self.union = None
        self.target_count = None
        self.correct = 0.0
        self.total = 0.0

    def _ensure_buffers(self, num_classes: int, device: torch.device):
        if self.num_classes is None:
            self.num_classes = num_classes
            self.intersection = torch.zeros(num_classes, dtype=torch.float64, device=device)
            self.union = torch.zeros(num_classes, dtype=torch.float64, device=device)
            self.target_count = torch.zeros(num_classes, dtype=torch.float64, device=device)
        elif self.num_classes != num_classes:
            raise ValueError(
                f'Number of classes changed during evaluation: '
                f'{self.num_classes} -> {num_classes}'
            )

    def _prepare_target(
        self,
        label_map: torch.Tensor,
        out_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if label_map.dim() == 4:
            if label_map.shape[1] != 1:
                raise ValueError(f'Expected label_map [B,H,W] or [B,1,H,W], got {tuple(label_map.shape)}')
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(f'Expected label_map [B,H,W] or [B,1,H,W], got {tuple(label_map.shape)}')

        label_map = label_map.long().to(device)
        if tuple(label_map.shape[-2:]) != tuple(out_hw):
            label_map = F.interpolate(
                label_map[:, None].float(),
                size=out_hw,
                mode='nearest',
            )[:, 0].long()
        return label_map

    def update(self, outputs: TensorDict, targets: TensorDict):
        if 'semantic_logits' not in outputs:
            raise ValueError('semantic_logits is required in semantic outputs.')
        if 'label_map' not in targets:
            raise ValueError('label_map is required in semantic targets.')

        logits = outputs['semantic_logits']            # [B, C, H, W]
        pred = logits.argmax(dim=1)                   # [B, H, W]
        target = self._prepare_target(
            label_map=targets['label_map'],
            out_hw=logits.shape[-2:],
            device=logits.device,
        )

        num_classes = logits.shape[1]
        self._ensure_buffers(num_classes=num_classes, device=logits.device)

        valid = target != self.ignore_index
        self.correct += float(((pred == target) & valid).sum().item())
        self.total += float(valid.sum().item())

        for cls_id in range(num_classes):
            pred_c = (pred == cls_id) & valid
            target_c = (target == cls_id) & valid

            inter = (pred_c & target_c).sum()
            union = (pred_c | target_c).sum()
            tgt_count = target_c.sum()

            self.intersection[cls_id] += inter.double()
            self.union[cls_id] += union.double()
            self.target_count[cls_id] += tgt_count.double()

    def compute(self) -> Dict[str, float]:
        if self.num_classes is None:
            return {}

        per_class_iou = self.intersection / self.union.clamp(min=1.0)
        per_class_acc = self.intersection / self.target_count.clamp(min=1.0)

        valid_iou = self.union > 0
        valid_acc = self.target_count > 0

        miou = per_class_iou[valid_iou].mean().item() if valid_iou.any() else 0.0
        macc = per_class_acc[valid_acc].mean().item() if valid_acc.any() else 0.0
        pixel_acc = self.correct / max(self.total, 1.0)

        out = {
            'semantic.miou': float(miou),
            'semantic.macc': float(macc),
            'semantic.pixel_acc': float(pixel_acc),
        }

        for i in range(self.num_classes):
            out[f'semantic.iou_class_{i}'] = float(per_class_iou[i].item())
            out[f'semantic.acc_class_{i}'] = float(per_class_acc[i].item())

        return out

def _round_up(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor) * divisor)


def _flip_image_batch(img_batch: torch.Tensor, flip_mode: str) -> torch.Tensor:
    if flip_mode == 'none':
        return img_batch
    if flip_mode == 'h':
        return torch.flip(img_batch, dims=[-1])
    if flip_mode == 'v':
        return torch.flip(img_batch, dims=[-2])
    if flip_mode == 'hv':
        return torch.flip(img_batch, dims=[-2, -1])
    raise ValueError(f'Unknown flip_mode: {flip_mode}')


def _deaugment_logits(logits: torch.Tensor, target_hw: tuple[int, int], flip_mode: str) -> torch.Tensor:
    if flip_mode == 'h':
        logits = torch.flip(logits, dims=[-1])
    elif flip_mode == 'v':
        logits = torch.flip(logits, dims=[-2])
    elif flip_mode == 'hv':
        logits = torch.flip(logits, dims=[-2, -1])
    elif flip_mode != 'none':
        raise ValueError(f'Unknown flip_mode: {flip_mode}')

    if tuple(logits.shape[-2:]) != tuple(target_hw):
        logits = F.interpolate(logits, size=target_hw, mode='bilinear', align_corners=False)
    return logits


@torch.no_grad()
def inference_with_tta(
    model,
    batch,
    tta_cfg: Optional[Dict],
):
    if tta_cfg is None or not bool(tta_cfg.get('enabled', False)):
        return model(batch)

    scales = [float(x) for x in tta_cfg.get('scales', [1.0])]
    flip_modes = list(tta_cfg.get('flip_modes', ['none']))
    size_divisor = int(tta_cfg.get('size_divisor', 14))

    base_img_batch = batch.img_batch
    target_hw = tuple(base_img_batch.shape[-2:])

    sum_4d: Dict[str, torch.Tensor] = {}
    sum_2d: Dict[str, torch.Tensor] = {}
    num_views = 0
    last_outputs = None

    for scale in scales:
        if scale <= 0:
            raise ValueError(f'Invalid TTA scale: {scale}')

        scaled_h = max(1, int(round(target_hw[0] * scale)))
        scaled_w = max(1, int(round(target_hw[1] * scale)))
        if size_divisor > 1:
            scaled_h = _round_up(scaled_h, size_divisor)
            scaled_w = _round_up(scaled_w, size_divisor)

        resized_img_batch = F.interpolate(
            base_img_batch,
            size=(scaled_h, scaled_w),
            mode='bilinear',
            align_corners=False,
        )

        for flip_mode in flip_modes:
            aug_batch = copy.deepcopy(batch)
            aug_batch.img_batch = _flip_image_batch(resized_img_batch, flip_mode)

            outputs = model(aug_batch)
            last_outputs = outputs

            for key, value in outputs.items():
                if not torch.is_tensor(value):
                    continue

                if value.dim() == 4:
                    deaug = _deaugment_logits(value, target_hw=target_hw, flip_mode=flip_mode)
                    if key not in sum_4d:
                        sum_4d[key] = deaug
                    else:
                        sum_4d[key] = sum_4d[key] + deaug

                elif value.dim() == 2:
                    if key not in sum_2d:
                        sum_2d[key] = value
                    else:
                        sum_2d[key] = sum_2d[key] + value

            num_views += 1

    if last_outputs is None or num_views == 0:
        raise RuntimeError('TTA produced no outputs.')

    merged_outputs = dict(last_outputs)
    for key, value in sum_4d.items():
        merged_outputs[key] = value / float(num_views)
    for key, value in sum_2d.items():
        merged_outputs[key] = value / float(num_views)

    return merged_outputs

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device: torch.device | str = 'cuda',
    visualizer: Optional[VisualizationManager] = None,
    epoch: Optional[int] = None,
    stage: str = 'val',
    tta_cfg: Optional[Dict] = None,
) -> Dict[str, float]:
    device = torch.device(device)
    model.eval()

    evaluator = MulticlassSemanticEvaluator()

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        outputs = inference_with_tta(model, batch, tta_cfg=tta_cfg)

        targets = {'label_map': batch.find_targets[0].semantic_label_map}
        evaluator.update(outputs, targets)

        if visualizer is not None:
            visualizer.save_semantic_batch(batch, outputs, targets, epoch=epoch, stage=stage)

    return evaluator.compute()