from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

from api.main import create_app
from berry_detector import video


class FakeDetector:
    device = torch.device("cpu")
    gpu_name = None

    def __init__(self, _config) -> None:
        pass

    def detect(self, _image: np.ndarray) -> np.ndarray:
        return np.array([[1.0, 2.0, 3.0, 0.0, 0.5, 0.0]])


class SingleFrameCapture:
    def __init__(self, _path: str) -> None:
        self.pending = True

    def isOpened(self) -> bool:
        return True

    def read(self):
        if not self.pending:
            return False, None
        self.pending = False
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


def test_endpoint_response_is_compatible(tmp_path: Path, monkeypatch) -> None:
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "sample.mp4").touch()
    monkeypatch.setattr(video.cv2, "VideoCapture", SingleFrameCapture)

    def fake_imwrite(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    monkeypatch.setattr(video.cv2, "imwrite", fake_imwrite)
    app = create_app(FakeDetector)

    with TestClient(app) as client:
        response = client.post(
            "/detector_task",
            json={
                "input_folder": str(input_folder),
                "output_folder": str(output_folder),
                "video_name": "sample",
            },
        )
        health = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Detección completada",
        "video_name": "sample",
        "output_folder": str(output_folder),
    }
    assert health.json()["status"] == "ready"


def test_missing_video_returns_404(tmp_path: Path) -> None:
    app = create_app(FakeDetector)
    with TestClient(app) as client:
        response = client.post(
            "/detector_task",
            json={
                "input_folder": str(tmp_path),
                "output_folder": str(tmp_path),
                "video_name": "missing",
            },
        )
    assert response.status_code == 404
