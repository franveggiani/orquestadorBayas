"""Interfaz de inferencia para un frame."""

from __future__ import annotations

import logging
from threading import Lock

import cv2
import numpy as np
import torch

from .architecture import BerryHourglassNet
from .checkpoint import load_checkpoint
from .config import DetectorConfig
from .decoding import decode_detections
from .geometry import get_affine_transform, transform_predictions


LOGGER = logging.getLogger(__name__)


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"Dispositivo no soportado: {requested}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE solicita CUDA pero PyTorch no detecta una GPU NVIDIA"
        )
    return device


class BerryDetector:
    """Carga una vez la red y detecta círculos en frames BGR."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.model = BerryHourglassNet(config.heads, num_stacks=2)
        self.checkpoint_epoch = load_checkpoint(self.model, config.model_path)
        self.model.to(self.device)
        self.model.eval()
        self._inference_lock = Lock()
        self.mean = np.asarray(config.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(config.std, dtype=np.float32).reshape(1, 1, 3)
        LOGGER.info(
            "Modelo cargado desde %s (epoch=%s) en %s",
            config.model_path,
            self.checkpoint_epoch,
            self.device,
        )

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[torch.Tensor, dict[str, np.ndarray | float | int]]:
        height, width = image.shape[:2]
        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        scale = max(height, width) * 1.0
        transform = get_affine_transform(
            center,
            scale,
            0,
            (self.config.input_width, self.config.input_height),
        )
        warped = cv2.warpAffine(
            image,
            transform,
            (self.config.input_width, self.config.input_height),
            flags=cv2.INTER_LINEAR,
        )
        normalized = ((warped / 255.0 - self.mean) / self.std).astype(
            np.float32
        )
        tensor = normalized.transpose(2, 0, 1).reshape(
            1,
            3,
            self.config.input_height,
            self.config.input_width,
        )
        metadata: dict[str, np.ndarray | float | int] = {
            "center": center,
            "scale": scale,
            "output_height": self.config.input_height // self.config.down_ratio,
            "output_width": self.config.input_width // self.config.down_ratio,
        }
        return torch.from_numpy(tensor), metadata

    def _postprocess(
        self,
        detections: torch.Tensor,
        metadata: dict[str, np.ndarray | float | int],
    ) -> np.ndarray:
        result = detections.detach().cpu().numpy().reshape(
            -1, detections.shape[2]
        )
        center = np.asarray(metadata["center"], dtype=np.float32)
        scale = float(metadata["scale"])
        output_height = int(metadata["output_height"])
        output_width = int(metadata["output_width"])
        result[:, :2] = transform_predictions(
            result[:, :2],
            center,
            scale,
            (output_width, output_height),
        )
        result[:, 2] *= scale / output_width
        return result.astype(np.float32, copy=False)

    def detect(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Se esperaba un frame BGR de tres canales")
        tensor, metadata = self._preprocess(image)
        tensor = tensor.to(self.device)
        with self._inference_lock, torch.inference_mode():
            output = self.model(tensor)[-1]
            detections = decode_detections(
                output["hm"].sigmoid_(),
                output["cl"],
                output["occ"],
                output["reg"],
                confidence_threshold=self.config.confidence_threshold,
                maximum=self.config.max_detections,
            )
        return self._postprocess(detections, metadata)

    @property
    def gpu_name(self) -> str | None:
        if self.device.type != "cuda":
            return None
        index = self.device.index if self.device.index is not None else 0
        return torch.cuda.get_device_name(index)
