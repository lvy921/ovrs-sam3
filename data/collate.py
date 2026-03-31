from __future__ import annotations

import math
from typing import Any, MutableMapping, Sequence

import torch
import torch.nn.functional as F

from ..models.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)

Sample = MutableMapping[str, Any]


def _pad_tensor_hw(x: torch.Tensor, out_h: int, out_w: int, value: float = 0.0) -> torch.Tensor:
    h, w = x.shape[-2:]
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), value=value)


def _round_up(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor) * divisor)


class OVSemanticCollator:
    def __init__(
        self,
        image_pad_value: float = 0.0,
        pad_size_divisor: int = 1,
        label_pad_value: int = 255,
    ):
        self.image_pad_value = float(image_pad_value)
        self.pad_size_divisor = int(pad_size_divisor)
        self.label_pad_value = int(label_pad_value)

    def _collate_images(self, samples: Sequence[Sample]):
        sizes = [(int(s['image'].shape[-2]), int(s['image'].shape[-1])) for s in samples]
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)

        if self.pad_size_divisor > 1:
            max_h = _round_up(max_h, self.pad_size_divisor)
            max_w = _round_up(max_w, self.pad_size_divisor)

        imgs = [_pad_tensor_hw(s['image'], max_h, max_w, self.image_pad_value) for s in samples]
        return torch.stack(imgs, dim=0), sizes, (max_h, max_w)

    def __call__(self, samples: Sequence[Sample]) -> BatchedDatapoint:
        samples = list(samples)
        if len(samples) == 0:
            raise ValueError('Empty batch.')

        img_batch, image_sizes, padded_hw = self._collate_images(samples)
        batch_size = len(samples)

        prompt_texts = []
        prompt_img_ids = []
        prompt_class_ids = []

        label_maps = []
        image_id_list = []
        original_size_list = []
        class_names = []
        class_counts = []

        for b, sample in enumerate(samples):
            texts = list(sample['class_texts'])
            class_names.append(texts)
            class_counts.append(len(texts))

            for class_id, text in enumerate(texts):
                prompt_texts.append(str(text))
                prompt_img_ids.append(b)
                prompt_class_ids.append(class_id)

            label_map = sample['label_map'].long()
            if tuple(label_map.shape[-2:]) != tuple(padded_hw):
                label_map = _pad_tensor_hw(label_map, padded_hw[0], padded_hw[1], self.label_pad_value).long()
            label_maps.append(label_map)

            image_id_list.append(int(sample.get('image_id', b)))
            orig_h, orig_w = sample.get('original_size', image_sizes[b])
            original_size_list.append(torch.tensor([orig_h, orig_w], dtype=torch.long))

        num_prompts = len(prompt_texts)

        find_stage = FindStage(
            img_ids=torch.tensor(prompt_img_ids, dtype=torch.long),
            text_ids=torch.arange(num_prompts, dtype=torch.long),
            input_boxes=torch.zeros((0, num_prompts, 4), dtype=torch.float32),
            input_boxes_mask=torch.zeros((num_prompts, 0), dtype=torch.bool),
            input_boxes_label=torch.zeros((0, num_prompts), dtype=torch.long),
            input_points=torch.zeros((0, num_prompts, 2), dtype=torch.float32),
            input_points_mask=torch.zeros((num_prompts, 0), dtype=torch.bool),
        )

        find_target = BatchedFindTarget(
            semantic_label_map=torch.stack(label_maps, dim=0),   # [B,H,W]
        )

        metadata = BatchedInferenceMetadata(
            original_image_id=torch.tensor(image_id_list, dtype=torch.long),
            original_size=torch.stack(original_size_list, dim=0),
            prompt_img_ids=torch.tensor(prompt_img_ids, dtype=torch.long),
            prompt_class_ids=torch.tensor(prompt_class_ids, dtype=torch.long),
            class_counts=torch.tensor(class_counts, dtype=torch.long),
            class_names=class_names,
        )

        raw_images = [s.get('raw_image') for s in samples] if any('raw_image' in s for s in samples) else None

        return BatchedDatapoint(
            img_batch=img_batch,
            find_text_batch=prompt_texts,
            find_inputs=[find_stage],
            find_targets=[find_target],
            find_metadatas=[metadata],
            raw_images=raw_images,
        )