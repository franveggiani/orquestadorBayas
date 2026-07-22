import torch
import pytest

from berry_detector.decoding import (
    _select_occlusion_candidates,
    decode_detections,
)


def test_vectorized_candidate_selection_matches_legacy_loop() -> None:
    scores = torch.tensor([[0.8, 0.4, 0.2]])
    indices = torch.tensor([[5, 2, 7]])
    occlusion = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    expected = torch.zeros_like(occlusion)
    for score, index in zip(scores.reshape(-1), indices.reshape(-1)):
        if score.item() > 0.4:
            expected.view(-1)[int(index.item())] = occlusion.view(-1)[
                int(index.item())
            ]

    selected = _select_occlusion_candidates(scores, indices, occlusion, 0.4)

    assert torch.equal(selected, expected)


def test_decoder_returns_legacy_circle_layout() -> None:
    heatmap = torch.zeros((1, 1, 4, 4))
    heatmap[0, 0, 1, 2] = 0.9
    radii = torch.full((1, 1, 4, 4), 3.0)
    occlusion = torch.zeros((1, 1, 4, 4))
    occlusion[0, 0, 1, 2] = 0.8
    regression = torch.zeros((1, 2, 4, 4))

    detections = decode_detections(
        heatmap,
        radii,
        occlusion,
        regression,
        confidence_threshold=0.4,
        maximum=4,
    )

    assert detections.shape == (1, 4, 6)
    assert detections[0, 0].tolist() == pytest.approx(
        [2.0, 1.0, 3.0, 0.8, 0.8, 0.0]
    )
