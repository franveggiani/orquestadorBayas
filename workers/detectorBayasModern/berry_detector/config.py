"""Configuración de inferencia del detector."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    """Parámetros fijos compatibles con el checkpoint histórico."""

    model_path: Path
    device: str = "cuda"
    input_height: int = 1024
    input_width: int = 576
    down_ratio: int = 4
    confidence_threshold: float = 0.4
    max_detections: int = 1000
    mean: tuple[float, float, float] = (0.408, 0.447, 0.470)
    std: tuple[float, float, float] = (0.289, 0.274, 0.278)

    @property
    def heads(self) -> dict[str, int]:
        return {"hm": 1, "cl": 1, "reg": 2, "occ": 1}

    @classmethod
    def from_env(cls) -> "DetectorConfig":
        return cls(
            model_path=Path(
                os.getenv(
                    "MODEL_PATH",
                    "/models/2022.11.30_grapes_mix_iou.pth",
                )
            ),
            device=os.getenv("DEVICE", "cuda"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.4")),
            max_detections=int(os.getenv("MAX_DETECTIONS", "1000")),
        )
