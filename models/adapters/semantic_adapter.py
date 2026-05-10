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
    def _extract_required_tensor(
        raw_outputs: Dict[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        value = raw_outputs.get(key, None)
        if value is None:
            raise ValueError(f"Raw outputs must contain '{key}'.")
        return value

    @staticmethod
    def _extract_optional_tensor(
        raw_outputs: Dict[str, torch.Tensor],
        key: str,
    ) -> Optional[torch.Tensor]:
        return raw_outputs.get(key, None)

    @staticmethod
    def _ensure_4d_map(
        x: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        if x.dim() == 5:
            if x.shape[2] != 1:
                raise ValueError(
                    f"Expected {key} as [B, C, 1, H, W] when 5D, "
                    f"got {tuple(x.shape)}."
                )
            x = x[:, :, 0]

        if x.dim() != 4:
            raise ValueError(
                f"Expected {key} as [B, C, H, W], got {tuple(x.shape)}."
            )

        return x

    @staticmethod
    def _ensure_3d_query(
        x: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected {key} as [B, C, D], got {tuple(x.shape)}."
            )
        return x

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
    def _validate_same_shape(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        lhs_key: str,
        rhs_key: str,
    ) -> None:
        if lhs.shape != rhs.shape:
            raise ValueError(
                f"Shape mismatch between {lhs_key} and {rhs_key}: "
                f"{tuple(lhs.shape)} vs {tuple(rhs.shape)}."
            )

    @staticmethod
    def _validate_query_shape(
        class_query: torch.Tensor,
        semantic_logits: torch.Tensor,
    ) -> None:
        if class_query.shape[:2] != semantic_logits.shape[:2]:
            raise ValueError(
                "Shape mismatch between class_query and semantic_logits: "
                f"class_query.shape[:2]={tuple(class_query.shape[:2])}, "
                f"semantic_logits.shape[:2]={tuple(semantic_logits.shape[:2])}."
            )

    def _build_chunk_train_outputs(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._ensure_4d_map(
            self._extract_required_tensor(raw_outputs, OUTPUT_KEYS.semantic_logits),
            OUTPUT_KEYS.semantic_logits,
        )
        class_query = self._ensure_3d_query(
            self._extract_required_tensor(raw_outputs, OUTPUT_KEYS.class_query),
            OUTPUT_KEYS.class_query,
        )

        actual_num_classes = int(semantic_logits.shape[1])
        expected_num_classes = self._infer_expected_num_classes(
            batch=batch,
            expected_num_classes=expected_num_classes,
        )
        self._validate_class_count(
            actual_num_classes=actual_num_classes,
            expected_num_classes=expected_num_classes,
        )
        self._validate_query_shape(
            class_query=class_query,
            semantic_logits=semantic_logits,
        )

        outputs = {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.class_query: class_query,
        }

        extra_token_aux_logits = self._extract_optional_tensor(
            raw_outputs,
            OUTPUT_KEYS.extra_token_aux_logits,
        )
        if extra_token_aux_logits is not None:
            outputs[OUTPUT_KEYS.extra_token_aux_logits] = self._ensure_4d_map(
                extra_token_aux_logits,
                OUTPUT_KEYS.extra_token_aux_logits,
            )

        return outputs

    def _build_final_outputs(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._ensure_4d_map(
            self._extract_required_tensor(raw_outputs, OUTPUT_KEYS.semantic_logits),
            OUTPUT_KEYS.semantic_logits,
        )
        final_logits = self._ensure_4d_map(
            self._extract_required_tensor(raw_outputs, OUTPUT_KEYS.final_logits),
            OUTPUT_KEYS.final_logits,
        )

        actual_num_classes = int(semantic_logits.shape[1])
        expected_num_classes = self._infer_expected_num_classes(
            batch=batch,
            expected_num_classes=expected_num_classes,
        )
        self._validate_class_count(
            actual_num_classes=actual_num_classes,
            expected_num_classes=expected_num_classes,
        )

        self._validate_same_shape(
            semantic_logits,
            final_logits,
            OUTPUT_KEYS.semantic_logits,
            OUTPUT_KEYS.final_logits,
        )

        semantic_score_map = semantic_logits.sigmoid()
        final_score_map = final_logits.sigmoid()
        final_pred = final_score_map.argmax(dim=1)

        outputs = {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.semantic_score_map: semantic_score_map,
            OUTPUT_KEYS.final_logits: final_logits,
            OUTPUT_KEYS.final_score_map: final_score_map,
            OUTPUT_KEYS.final_pred: final_pred,
        }

        class_query = self._extract_optional_tensor(
            raw_outputs,
            OUTPUT_KEYS.class_query,
        )
        if class_query is not None:
            class_query = self._ensure_3d_query(
                class_query,
                OUTPUT_KEYS.class_query,
            )
            self._validate_query_shape(
                class_query=class_query,
                semantic_logits=semantic_logits,
            )
            outputs[OUTPUT_KEYS.class_query] = class_query

        extra_token_aux_logits = self._extract_optional_tensor(
            raw_outputs,
            OUTPUT_KEYS.extra_token_aux_logits,
        )
        if extra_token_aux_logits is not None:
            outputs[OUTPUT_KEYS.extra_token_aux_logits] = self._ensure_4d_map(
                extra_token_aux_logits,
                OUTPUT_KEYS.extra_token_aux_logits,
            )

        return outputs

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int] = None,
        output_mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        output_mode = str(output_mode)

        if output_mode == "train":
            return self._build_chunk_train_outputs(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=expected_num_classes,
            )

        if output_mode in {"final", "infer"}:
            return self._build_final_outputs(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=expected_num_classes,
            )

        raise ValueError(
            f"Unknown output_mode={output_mode}. "
            "Supported modes are: 'train', 'final', 'infer'."
        )


class HybridSegAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int] = None,
        output_mode: str = "train",
    ):
        raise NotImplementedError(
            "HybridSegAdapter is not implemented yet."
        )