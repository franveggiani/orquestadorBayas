import gc
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from berry_detector.architecture import BerryHourglassNet


HEADS = {"hm": 1, "cl": 1, "reg": 2, "occ": 1}


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1",
    reason="la comparación construye dos modelos Hourglass grandes",
)
def test_state_dict_matches_legacy_hourglass() -> None:
    legacy_path = (
        Path(__file__).parents[2]
        / "detectorBayas"
        / "src/lib/models/networks/large_hourglass.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_hourglass", legacy_path)
    assert spec is not None and spec.loader is not None
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)

    torch.manual_seed(42)
    legacy = legacy_module.get_large_hourglass_net(0, HEADS, 0).eval()
    legacy_shapes = {
        key: tuple(value.shape) for key, value in legacy.state_dict().items()
    }
    with torch.inference_mode():
        legacy_outputs = [
            {key: value.clone() for key, value in stack.items()}
            for stack in legacy(torch.zeros(1, 3, 64, 64))
        ]
    del legacy
    gc.collect()

    torch.manual_seed(42)
    modern = BerryHourglassNet(HEADS, num_stacks=2).eval()
    modern_shapes = {
        key: tuple(value.shape) for key, value in modern.state_dict().items()
    }
    assert modern_shapes == legacy_shapes
    with torch.inference_mode():
        modern_outputs = modern(torch.zeros(1, 3, 64, 64))
    for legacy_stack, modern_stack in zip(legacy_outputs, modern_outputs):
        for head in HEADS:
            torch.testing.assert_close(
                modern_stack[head], legacy_stack[head], rtol=0, atol=0
            )
