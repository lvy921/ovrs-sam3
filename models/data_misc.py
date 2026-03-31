from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch


MyTensor = Union[torch.Tensor, List[Any]]


@dataclass
class FindStage:
    img_ids: MyTensor
    img_ids__type = torch.long

    text_ids: MyTensor
    text_ids__type = torch.long

    input_boxes: MyTensor
    input_boxes__type = torch.float

    input_boxes_mask: MyTensor
    input_boxes_mask__type = torch.bool

    input_boxes_label: MyTensor
    input_boxes_label__type = torch.long

    input_points: MyTensor
    input_points__type = torch.float

    input_points_mask: MyTensor
    input_points_mask__type = torch.bool


@dataclass
class BatchedFindTarget:
    semantic_label_map: MyTensor
    semantic_label_map__type = torch.long


@dataclass
class BatchedInferenceMetadata:
    original_image_id: MyTensor
    original_image_id__type = torch.long

    original_size: MyTensor
    original_size__type = torch.long

    prompt_img_ids: MyTensor
    prompt_img_ids__type = torch.long

    prompt_class_ids: MyTensor
    prompt_class_ids__type = torch.long

    class_counts: MyTensor
    class_counts__type = torch.long

    class_names: List[List[str]]


@dataclass
class BatchedDatapoint:
    img_batch: torch.Tensor
    find_text_batch: List[str]
    find_inputs: List[FindStage]
    find_targets: List[BatchedFindTarget]
    find_metadatas: List[BatchedInferenceMetadata]
    raw_images: Optional[List[Any]] = None