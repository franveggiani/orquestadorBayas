"""Procesamiento de videos y persistencia compatible con el pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Protocol

import cv2
import numpy as np


class FrameDetector(Protocol):
    def detect(self, image: np.ndarray) -> np.ndarray: ...


class VideoProcessingError(RuntimeError):
    """Error de entrada/salida durante el procesamiento del video."""


class VideoNotFoundError(VideoProcessingError):
    """No existe el MP4 solicitado."""


class VideoUnreadableError(VideoProcessingError):
    """OpenCV no pudo abrir o leer el video."""


class OutputWriteError(VideoProcessingError):
    """No fue posible escribir un artefacto de salida."""


def _draw_detections(
    frame: np.ndarray, detections: np.ndarray
) -> np.ndarray:
    """Dibuja círculos y oclusiones sin modificar el frame de entrada."""
    annotated = frame.copy()
    for detection in detections:
        x, y, radius = detection[:3]
        if not np.isfinite((x, y, radius)).all() or radius <= 0:
            continue

        center = (int(round(float(x))), int(round(float(y))))
        cv2.circle(
            annotated,
            center,
            max(1, int(round(float(radius)))),
            (0, 255, 0),
            2,
        )
        if detection.shape[0] > 3:
            cv2.putText(
                annotated,
                f"{float(detection[3]):.2f}",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return annotated


def _atomic_write_json(data: dict[int, dict[int, list[float]]],
                       destination: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(data, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise OutputWriteError(
            f"No se pudo escribir el JSON {destination}: {exc}"
        ) from exc


def process_video(
    detector: FrameDetector,
    input_folder: str,
    output_folder: str,
    video_name: str,
) -> Path:
    video_path = Path(input_folder) / f"{video_name}.mp4"
    if not video_path.is_file():
        raise VideoNotFoundError(f"No existe el video: {video_path}")

    output_path = Path(output_folder)
    frames_path = output_path / "detector_frames"
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        frames_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputWriteError(
            f"No se pudo crear el directorio de salida {output_path}: {exc}"
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoUnreadableError(f"OpenCV no pudo abrir {video_path}")

    detections_by_frame: dict[int, dict[int, list[float]]] = {}
    frame_number = 0
    try:
        while True:
            readable, frame = capture.read()
            if not readable:
                break

            detections = detector.detect(frame).astype(float, copy=False)
            valid = detections[
                (detections[:, 0] > 0) & (detections[:, 1] > 0)
            ]
            annotated_frame = _draw_detections(frame, valid)
            frame_path = frames_path / f"{frame_number:05d}.jpg"
            if not cv2.imwrite(str(frame_path), annotated_frame):
                raise OutputWriteError(
                    f"No se pudo escribir el frame {frame_number} en {frame_path}"
                )
            detections_by_frame[frame_number] = {
                index: [float(value) for value in row[:3]]
                for index, row in enumerate(valid)
            }
            frame_number += 1
    finally:
        capture.release()

    if frame_number == 0:
        raise VideoUnreadableError(f"El video no contiene frames legibles: {video_path}")

    destination = output_path / f"{video_name}.json"
    _atomic_write_json(detections_by_frame, destination)
    return destination
