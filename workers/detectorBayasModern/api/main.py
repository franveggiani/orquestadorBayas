"""Servicio FastAPI compatible con detectorBayas."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
import torch

from berry_detector import BerryDetector, DetectorConfig
from berry_detector.checkpoint import CheckpointError
from berry_detector.video import (
    OutputWriteError,
    VideoNotFoundError,
    VideoUnreadableError,
    process_video,
)
from .schemas import DetectorRequest


LOGGER = logging.getLogger(__name__)
DetectorFactory = Callable[[DetectorConfig], BerryDetector]


def create_app(detector_factory: DetectorFactory = BerryDetector) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = DetectorConfig.from_env()
        app.state.detector = detector_factory(config)
        app.state.processing_lock = Lock()
        yield

    application = FastAPI(lifespan=lifespan)

    @application.get("/health")
    def health(request: Request) -> dict[str, object]:
        detector = request.app.state.detector
        return {
            "status": "ready",
            "model_loaded": True,
            "device": str(detector.device),
            "gpu_name": detector.gpu_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }

    @application.post("/detector_task")
    def detector_task(
        detector_request: DetectorRequest, request: Request
    ) -> dict[str, str]:
        try:
            with request.app.state.processing_lock:
                process_video(
                    request.app.state.detector,
                    detector_request.input_folder,
                    detector_request.output_folder,
                    detector_request.video_name,
                )
        except VideoNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except VideoUnreadableError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OutputWriteError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Falló la detección de %s", detector_request.video_name)
            raise HTTPException(
                status_code=500, detail="Falló la inferencia del detector"
            ) from exc

        return {
            "message": "Detección completada",
            "video_name": detector_request.video_name,
            "output_folder": detector_request.output_folder,
        }

    return application


app = create_app()


__all__ = ["app", "create_app", "CheckpointError"]
