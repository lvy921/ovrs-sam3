from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..data_misc import BatchedDatapoint


class QueryMaskSemanticAdapter(nn.Module):
    """
    Convert prompt-expanded SAM3 query outputs into multi-class semantic logits.

    New semantic flow:
        raw_outputs (per prompt, per query)
            -> aggregate queries inside each prompt
            -> prompt_logits: [P, 1, H, W]
            -> scatter back to image/class grid
            -> semantic_logits: [B, C, H, W]

    Where:
        B = number of images in batch
        C = number of classes per image
        P = total number of prompt-expanded text queries in batch
    """

    def __init__(self, topk: Optional[int] = None, aggregation: str = "weighted_sum"):
        super().__init__()
        self.topk = topk
        self.aggregation = aggregation

    def _select_topk(
        self,
        query_scores: torch.Tensor,   # [P, Q]
        query_masks: torch.Tensor,    # [P, Q, H, W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.topk is None:
            return query_scores, query_masks

        num_queries = query_scores.shape[1]
        k = min(int(self.topk), int(num_queries))
        topk_scores, topk_idx = torch.topk(query_scores, k=k, dim=1)

        gather_idx = topk_idx[..., None, None].expand(
            -1, -1, query_masks.size(-2), query_masks.size(-1)
        )
        topk_masks = torch.gather(query_masks, dim=1, index=gather_idx)
        return topk_scores, topk_masks

    def _aggregate_prompt_logits(self, pred_logits, pred_masks) -> torch.Tensor:
        query_scores = pred_logits.squeeze(-1).sigmoid()
        query_scores, pred_masks = self._select_topk(query_scores, pred_masks)

        if self.aggregation == "max":
            weighted_masks = pred_masks * query_scores[..., None, None]
            prompt_logits = weighted_masks.max(dim=1, keepdim=True).values
        elif self.aggregation == "logsumexp":
            weighted_masks = pred_masks + query_scores[..., None, None].log().clamp(min=-20.0)
            prompt_logits = torch.logsumexp(weighted_masks, dim=1, keepdim=True)
        elif self.aggregation == "weighted_sum":
            weights = query_scores / query_scores.sum(dim=1, keepdim=True).clamp(min=1e-6)
            prompt_logits = (pred_masks * weights[..., None, None]).sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return prompt_logits

    @staticmethod
    def _build_semantic_logits(
        prompt_logits: torch.Tensor,      # [P, 1, H, W]
        batch: BatchedDatapoint,
    ) -> torch.Tensor:
        if len(batch.find_metadatas) != 1:
            raise ValueError("Current semantic adapter assumes exactly one metadata stage per batch.")

        meta = batch.find_metadatas[0]
        if meta.prompt_img_ids is None or meta.prompt_class_ids is None:
            raise ValueError("prompt_img_ids and prompt_class_ids are required for multi-class semantic assembly.")

        prompt_img_ids = meta.prompt_img_ids.to(prompt_logits.device).long()      # [P]
        prompt_class_ids = meta.prompt_class_ids.to(prompt_logits.device).long()  # [P]

        batch_size = int(batch.img_batch.shape[0])
        if meta.class_counts is not None:
            num_classes = int(meta.class_counts.max().item())
        else:
            num_classes = int(prompt_class_ids.max().item()) + 1

        h, w = prompt_logits.shape[-2:]
        semantic_logits = prompt_logits.new_full((batch_size, num_classes, h, w), -20.0)

        # prompt_logits[:, 0] -> [P, H, W]
        semantic_logits[prompt_img_ids, prompt_class_ids] = prompt_logits[:, 0]
        return semantic_logits

    def forward(self, raw_outputs, batch):
        pred_masks = raw_outputs["pred_masks"]
        pred_logits = raw_outputs["pred_logits"]

        if pred_masks.dim() == 5:
            pred_masks = pred_masks[-1]
        if pred_logits.dim() == 4:
            pred_logits = pred_logits[-1]

        prompt_logits = self._aggregate_prompt_logits(pred_logits, pred_masks)
        semantic_logits = self._build_semantic_logits(prompt_logits, batch)
        return {"semantic_logits": semantic_logits}