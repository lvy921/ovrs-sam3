from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .adapters.semantic_adapter import QueryMaskSemanticAdapter
from .sam3_core import Sam3Core
from .segmentor import SAM3Segmentor


def convert_sam3_image_to_core(model: nn.Module) -> Sam3Core:
    model.__class__ = Sam3Core
    model.matcher = None
    return model


def build_segmentor_from_sam3_image(
    sam3_image_model: nn.Module,
    semantic_topk: Optional[int] = 20,
    semantic_aggregation: str = 'weighted_sum',
) -> SAM3Segmentor:
    core = convert_sam3_image_to_core(sam3_image_model)
    return SAM3Segmentor(
        core=core,
        semantic_adapter=QueryMaskSemanticAdapter(
            topk=semantic_topk,
            aggregation=semantic_aggregation,
        ),
    )