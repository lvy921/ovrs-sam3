from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


Sample = MutableMapping[str, Any]


def _to_tensor_image(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            return image.float()
        raise ValueError(f'Unsupported image tensor shape: {tuple(image.shape)}')
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = image[..., None]
        if image.ndim != 3:
            raise ValueError(f'Unsupported image array shape: {image.shape}')
        tensor = torch.from_numpy(image)
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        return tensor.float()
    raise TypeError(f'Unsupported image type: {type(image)}')


def _to_tensor_mask(mask: Any) -> torch.Tensor:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        return mask
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    if isinstance(mask, np.ndarray):
        return torch.from_numpy(mask)
    raise TypeError(f'Unsupported mask type: {type(mask)}')


def _resize_tensor_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if image.ndim == 3:
        image = image[None]
        out = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
        return out[0]
    raise ValueError(f'Unsupported image shape: {tuple(image.shape)}')


def _resize_label_map(label_map: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if label_map is None:
        return None
    if label_map.ndim != 2:
        raise ValueError(f'Unsupported label_map shape: {tuple(label_map.shape)}')
    label_map = label_map[None, None].float()
    label_map = F.interpolate(label_map, size=size, mode='nearest')[0, 0]
    return label_map.long()

class Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    """Convert common image / mask containers into torch tensors.

    Expected keys (all optional except ``image``):
    - image
    - semantic_mask
    - instance_masks
    - boxes
    """

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        sample['image'] = _to_tensor_image(sample['image'])
        if 'label_map' in sample:
            sample['label_map'] = _to_tensor_mask(sample.get('label_map')).long()
        return sample


class ConvertImageDtype:
    def __init__(self, dtype: torch.dtype = torch.float32, scale: bool = True):
        self.dtype = dtype
        self.scale = bool(scale)

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        image = sample['image']
        if not isinstance(image, torch.Tensor):
            raise TypeError('ConvertImageDtype expects image to be a torch.Tensor')
        image = image.to(self.dtype)
        if self.scale and image.max() > 1.0:
            image = image / 255.0
        sample['image'] = image
        return sample


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        image = sample['image']
        sample['image'] = (image - self.mean.to(image.device, image.dtype)) / self.std.to(image.device, image.dtype)
        return sample


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = tuple(size)

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        image = sample['image']
        h, w = image.shape[-2:]
        out_h, out_w = self.size

        sample['image'] = _resize_tensor_image(image, (out_h, out_w))

        if 'label_map' in sample and sample['label_map'] is not None:
            sample['label_map'] = _resize_label_map(sample['label_map'], (out_h, out_w))

        sample['img_shape'] = (out_h, out_w)
        sample['scale_factor'] = (out_w / w, out_h / h)
        return sample


class ResizeLongestSide:
    def __init__(self, long_side: int):
        self.long_side = int(long_side)

    def __call__(self, sample: Sample) -> Sample:
        image = sample['image']
        h, w = image.shape[-2:]
        scale = self.long_side / max(h, w)
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
        return Resize((out_h, out_w))(sample)

class RandomResize:
    def __init__(self, scales: Sequence[Tuple[int, int]]):
        self.scales = list(scales)

    def __call__(self, sample: Sample) -> Sample:
        size = random.choice(self.scales)
        return Resize(size)(sample)


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample
        sample = dict(sample)
        sample['image'] = torch.flip(sample['image'], dims=[-1])

        if 'label_map' in sample and sample['label_map'] is not None:
            sample['label_map'] = torch.flip(sample['label_map'], dims=[-1])

        return sample


class PadToSize:
    def __init__(self, size: Tuple[int, int], image_pad_value: float = 0.0, label_pad_value: int = 255):
        self.size = tuple(size)
        self.image_pad_value = float(image_pad_value)
        self.label_pad_value = int(label_pad_value)

    def _pad_last_two_dims(self, x: torch.Tensor, pad_value: float) -> torch.Tensor:
        out_h, out_w = self.size
        h, w = x.shape[-2:]
        pad_h = max(0, out_h - h)
        pad_w = max(0, out_w - w)
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), value=pad_value)

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        sample['image'] = self._pad_last_two_dims(sample['image'], self.image_pad_value)

        if 'label_map' in sample and sample['label_map'] is not None:
            sample['label_map'] = self._pad_last_two_dims(sample['label_map'], self.label_pad_value).long()

        sample['pad_shape'] = self.size
        return sample

