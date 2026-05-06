from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .data_misc import BatchedDatapoint
from .sam3_image import Sam3Image
from .task_modes import OUTPUT_KEYS, normalize_task_mode


class SAM3Segmentor(nn.Module):
    def __init__(
        self,
        core: Sam3Image,
        adapter: nn.Module,
        task_mode: str,
    ):
        super().__init__()
        self.core = core
        self.adapter = adapter
        self.task_mode = normalize_task_mode(task_mode)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def clear_text_cache(self) -> None:
        self.core.clear_text_cache()

    def prepare_text_cache(
        self,
        class_names: List[str],
        device: Optional[torch.device] = None,
        force: bool = False,
    ) -> None:
        self.core.prepare_text_cache(
            class_texts=class_names,
            device=device,
            force=force,
        )

    @staticmethod
    def _build_mixer_cache_item(
        outputs: Dict[str, torch.Tensor],
        chunk_class_ids: List[int],
        detach_score_maps: bool,
    ) -> Dict[str, torch.Tensor | list[int]]:
        required_keys = (
            OUTPUT_KEYS.semantic_logits,
            OUTPUT_KEYS.clip_dense_logits,
            OUTPUT_KEYS.class_query,
        )

        for key in required_keys:
            if key not in outputs:
                raise ValueError(
                    f"Chunk outputs must contain '{key}' for final mixer."
                )

        semantic_logits = outputs[OUTPUT_KEYS.semantic_logits]
        clip_dense_logits = outputs[OUTPUT_KEYS.clip_dense_logits]
        class_query = outputs[OUTPUT_KEYS.class_query]

        if detach_score_maps:
            semantic_logits = semantic_logits.detach()
            clip_dense_logits = clip_dense_logits.detach()

        return {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.clip_dense_logits: clip_dense_logits,
            OUTPUT_KEYS.class_query: class_query,
            "chunk_class_ids": list(chunk_class_ids),
        }

    def iter_chunk_outputs(
        self,
        batch: BatchedDatapoint,
    ) -> Iterator[Dict[str, Any]]:
        for chunk in self.core.iter_chunk_raw_outputs(batch):
            raw_outputs = chunk["raw_outputs"]
            chunk_class_ids = chunk["chunk_class_ids"]

            train_outputs = self.adapter(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=len(chunk_class_ids),
                output_mode="train",
            )

            yield {
                "chunk_start": chunk["chunk_start"],
                "chunk_end": chunk["chunk_end"],
                "chunk_class_ids": chunk_class_ids,
                "chunk_class_names": chunk["chunk_class_names"],
                "raw_outputs": raw_outputs,
                "train_outputs": train_outputs,
            }

    def run_final_mixer_from_chunks(
        self,
        mixer_cache: List[Dict[str, torch.Tensor | list[int]]],
        batch: Optional[BatchedDatapoint] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.core.run_final_mixer_from_chunks(
            mixer_cache=mixer_cache,
            batch=batch,
        )

    def forward(self, batch: BatchedDatapoint) -> dict[str, torch.Tensor]:
        mixer_cache = []
        extra_token_aux_chunks = []

        for chunk in self.core.iter_chunk_raw_outputs(batch):
            raw_outputs = chunk["raw_outputs"]

            chunk_outputs = self.adapter(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=len(chunk["chunk_class_ids"]),
                output_mode="train",
            )

            mixer_cache.append(
                self._build_mixer_cache_item(
                    outputs=chunk_outputs,
                    chunk_class_ids=chunk["chunk_class_ids"],
                    detach_score_maps=False,
                )
            )

            if OUTPUT_KEYS.extra_token_aux_logits in chunk_outputs:
                extra_token_aux_chunks.append(
                    chunk_outputs[OUTPUT_KEYS.extra_token_aux_logits]
                )

        final_raw_outputs = self.run_final_mixer_from_chunks(
            mixer_cache=mixer_cache,
            batch=batch,
        )

        if len(extra_token_aux_chunks) > 0:
            final_raw_outputs[OUTPUT_KEYS.extra_token_aux_logits] = torch.cat(
                extra_token_aux_chunks,
                dim=1,
            )

        return self.adapter(
            raw_outputs=final_raw_outputs,
            batch=batch,
            expected_num_classes=None,
            output_mode="infer",
        )