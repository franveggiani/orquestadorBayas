import json
from pathlib import Path

import numpy as np

from berry_detector import video


class FakeCapture:
    def __init__(self, _path: str) -> None:
        self.frames = [
            np.full((8, 12, 3), value, dtype=np.uint8)
            for value in (10, 20)
        ]
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeDetector:
    def detect(self, _image: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [6.0, 4.0, 2.0, 0.0, 0.8, 0.0],
                [-1.0, 5.0, 4.0, 0.0, 0.7, 0.0],
            ],
            dtype=np.float32,
        )


def test_process_video_preserves_json_and_frame_contract(
    tmp_path: Path, monkeypatch
) -> None:
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "sample.mp4").touch()
    monkeypatch.setattr(video.cv2, "VideoCapture", FakeCapture)

    written_frames: list[np.ndarray] = []

    def fake_imwrite(path: str, frame: np.ndarray) -> bool:
        written_frames.append(frame.copy())
        Path(path).write_bytes(b"jpeg")
        return True

    monkeypatch.setattr(video.cv2, "imwrite", fake_imwrite)

    destination = video.process_video(
        FakeDetector(), str(input_folder), str(output_folder), "sample"
    )

    assert destination == output_folder / "sample.json"
    assert sorted(path.name for path in (output_folder / "detector_frames").iterdir()) == [
        "00000.jpg",
        "00001.jpg",
    ]
    assert json.loads(destination.read_text()) == {
        "0": {"0": [6.0, 4.0, 2.0]},
        "1": {"0": [6.0, 4.0, 2.0]},
    }
    assert len(written_frames) == 2
    assert np.any(written_frames[0] != 10)
    assert np.any(written_frames[1] != 20)
    assert not list(output_folder.glob(".*.tmp"))


def test_draw_detections_does_not_modify_original_frame() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    detections = np.array(
        [[20.0, 20.0, 8.0, 0.25, 0.9, 0.0]], dtype=np.float32
    )

    annotated = video._draw_detections(frame, detections)

    assert not np.any(frame)
    assert np.any(annotated[:, :, 1] > 0)
