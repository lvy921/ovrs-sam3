from __future__ import annotations

import torch
import torch.nn as nn

from .adapters.semantic_adapter import QueryMaskSemanticAdapter
from .data_misc import BatchedDatapoint
from .sam3_core import Sam3Core


class SAM3Segmentor(nn.Module):
    def __init__(
        self,
        core: Sam3Core,
        semantic_adapter: nn.Module | None = None,
    ):
        super().__init__()
        self.core = core
        self.semantic_adapter = semantic_adapter or QueryMaskSemanticAdapter()

    def forward(self, batch: BatchedDatapoint) -> dict[str, torch.Tensor]:
        raw_outputs = self.core(batch)
        return self.semantic_adapter(raw_outputs, batch=batch)