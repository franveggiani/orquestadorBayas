from pathlib import Path

import pytest
import torch
from torch import nn

from berry_detector.checkpoint import CheckpointError, load_checkpoint


def _model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(2, 2))


def test_loads_data_parallel_checkpoint_strictly(tmp_path: Path) -> None:
    source = _model()
    checkpoint = {
        "epoch": 7,
        "state_dict": {
            f"module.{key}": value.clone()
            for key, value in source.state_dict().items()
        },
    }
    path = tmp_path / "model.pth"
    torch.save(checkpoint, path)

    target = _model()
    epoch = load_checkpoint(target, path)

    assert epoch == 7
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


def test_rejects_incompatible_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "bad.pth"
    torch.save({"state_dict": {"unexpected": torch.ones(1)}}, path)

    with pytest.raises(CheckpointError, match="faltan claves"):
        load_checkpoint(_model(), path)


def test_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="No existe"):
        load_checkpoint(_model(), tmp_path / "missing.pth")
