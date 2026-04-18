from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..data_misc import BatchedDatapoint
from ..task_modes import OUTPUT_KEYS


class SemanticSegAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _extract_semantic_logits(
        raw_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        semantic_logits = raw_outputs.get(OUTPUT_KEYS.semantic_logits, None)
        if semantic_logits is None:
            raise ValueError(
                f"Raw outputs must contain '{OUTPUT_KEYS.semantic_logits}'."
            )

        if semantic_logits.dim() == 5:
            if semantic_logits.shape[2] != 1:
                raise ValueError(
                    f"Expected semantic_logits as [B, C, 1, H, W], got {tuple(semantic_logits.shape)}"
                )
            semantic_logits = semantic_logits[:, :, 0]
        elif semantic_logits.dim() != 4:
            raise ValueError(
                f"Expected semantic_logits as [B, C, H, W] or [B, C, 1, H, W], got {tuple(semantic_logits.shape)}"
            )

        return semantic_logits.contiguous()

    @staticmethod
    def _extract_presence_score(
        raw_outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        presence_score = raw_outputs.get("presence_score", None)
        if presence_score is None:
            return None

        if presence_score.dim() != 2:
            raise ValueError(
                f"Expected presence_score as [B, C], got {tuple(presence_score.shape)}"
            )

        return presence_score.contiguous()

    @staticmethod
    def _infer_expected_num_classes(
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Optional[int]:
        if expected_num_classes is not None:
            return int(expected_num_classes)

        if len(batch.find_metadatas) == 0:
            return None

        try:
            return int(batch.find_metadatas[0].num_classes)
        except Exception:
            return None

    @staticmethod
    def _validate_class_count(
        actual_num_classes: int,
        expected_num_classes: Optional[int],
    ) -> None:
        if expected_num_classes is None:
            return

        if actual_num_classes != int(expected_num_classes):
            raise ValueError(
                f"Class count mismatch: expected {expected_num_classes}, "
                f"but got {actual_num_classes} channels."
            )

    @staticmethod
    def _validate_presence_shape(
        semantic_logits: torch.Tensor,
        presence_score: Optional[torch.Tensor],
    ) -> None:
        if presence_score is None:
            return

        bsz, num_classes = semantic_logits.shape[:2]
        if tuple(presence_score.shape) != (bsz, num_classes):
            raise ValueError(
                f"presence_score shape mismatch: expected {(bsz, num_classes)}, "
                f"got {tuple(presence_score.shape)}"
            )

    @staticmethod
    def _apply_presence_to_logits(
        semantic_logits: torch.Tensor,
        presence_score: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if presence_score is None:
            return semantic_logits

        presence_score_4d = presence_score[:, :, None, None]
        final_logits = semantic_logits * presence_score_4d
        return final_logits.contiguous()

    def _build_train_outputs(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._extract_semantic_logits(raw_outputs)
        presence_score = self._extract_presence_score(raw_outputs)

        actual_num_classes = int(semantic_logits.shape[1])
        expected_num_classes = self._infer_expected_num_classes(
            batch=batch,
            expected_num_classes=expected_num_classes,
        )
        self._validate_class_count(
            actual_num_classes=actual_num_classes,
            expected_num_classes=expected_num_classes,
        )
        self._validate_presence_shape(
            semantic_logits=semantic_logits,
            presence_score=presence_score,
        )

        final_logits = self._apply_presence_to_logits(
            semantic_logits=semantic_logits,
            presence_score=presence_score,
        )

        outputs = {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.final_score_map: final_logits,
        }
        if presence_score is not None:
            outputs["presence_score"] = presence_score

        return outputs

    def _build_inference_outputs(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._extract_semantic_logits(raw_outputs)
        presence_score = self._extract_presence_score(raw_outputs)

        actual_num_classes = int(semantic_logits.shape[1])
        expected_num_classes = self._infer_expected_num_classes(
            batch=batch,
            expected_num_classes=expected_num_classes,
        )
        self._validate_class_count(
            actual_num_classes=actual_num_classes,
            expected_num_classes=expected_num_classes,
        )
        self._validate_presence_shape(
            semantic_logits=semantic_logits,
            presence_score=presence_score,
        )

        final_logits = self._apply_presence_to_logits(
            semantic_logits=semantic_logits,
            presence_score=presence_score,
        )
        final_pred = final_logits.argmax(dim=1)

        outputs = {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.semantic_score_map: semantic_logits,
            OUTPUT_KEYS.final_score_map: final_logits,
            OUTPUT_KEYS.final_pred: final_pred,
        }
        if presence_score is not None:
            outputs["presence_score"] = presence_score

        return outputs

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int] = None,
        output_mode: str = "both",
    ) -> Dict[str, Dict[str, torch.Tensor]] | Dict[str, torch.Tensor]:
        output_mode = str(output_mode)

        if output_mode == "train":
            return self._build_train_outputs(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=expected_num_classes,
            )

        if output_mode == "infer":
            return self._build_inference_outputs(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=expected_num_classes,
            )

        if output_mode == "both":
            train_outputs = self._build_train_outputs(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=expected_num_classes,
            )

            inference_outputs = self._build_inference_outputs(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=expected_num_classes,
            )

            return {
                "train_outputs": train_outputs,
                "inference_outputs": inference_outputs,
            }

        raise ValueError(
            f"Unknown output_mode={output_mode}. "
            "Supported modes are: 'train', 'infer', 'both'."
        )


class HybridSegAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int] = None,
        output_mode: str = "both",
    ):
        raise NotImplementedError(
            "HybridSegAdapter is not implemented yet."
        )