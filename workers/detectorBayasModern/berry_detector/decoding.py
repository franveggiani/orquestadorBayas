"""Decoder compatible con CircleNet CDIou usado en producción."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _gather(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    dimension = features.size(2)
    expanded = indices.unsqueeze(2).expand(
        indices.size(0), indices.size(1), dimension
    )
    return features.gather(1, expanded)


def _transpose_and_gather(
    features: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    features = features.permute(0, 2, 3, 1).contiguous()
    features = features.view(features.size(0), -1, features.size(3))
    return _gather(features, indices)


def _nms(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    padding = (kernel - 1) // 2
    maximum = F.max_pool2d(
        heatmap, (kernel, kernel), stride=1, padding=padding
    )
    return heatmap * (maximum == heatmap).float()


def _topk(
    scores: torch.Tensor, maximum: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, categories, height, width = scores.size()
    category_scores, category_indices = torch.topk(
        scores.view(batch, categories, -1), maximum
    )
    category_indices %= height * width
    ys = torch.div(category_indices, width, rounding_mode="floor").float()
    xs = (category_indices % width).float()

    top_scores, top_indices = torch.topk(
        category_scores.view(batch, -1), maximum
    )
    classes = torch.div(top_indices, maximum, rounding_mode="floor").int()
    gathered_indices = _gather(
        category_indices.view(batch, -1, 1), top_indices
    ).view(batch, maximum)
    ys = _gather(ys.view(batch, -1, 1), top_indices).view(batch, maximum)
    xs = _gather(xs.view(batch, -1, 1), top_indices).view(batch, maximum)
    return top_scores, gathered_indices, classes, ys, xs


def _select_occlusion_candidates(
    scores: torch.Tensor,
    indices: torch.Tensor,
    occlusion: torch.Tensor,
    confidence_threshold: float,
) -> torch.Tensor:
    """Copia oclusiones seleccionadas sin sincronizar una vez por candidato."""
    selected = torch.zeros_like(occlusion)
    # El legado compara ``float32.item()`` contra un float de Python. Promover
    # antes de comparar conserva ese detalle en valores exactamente al umbral.
    keep = scores[0].double() > confidence_threshold
    kept_indices = indices[0, keep]
    selected.view(-1)[kept_indices] = occlusion.view(-1)[kept_indices]
    return selected


def decode_detections(
    heatmap: torch.Tensor,
    radius: torch.Tensor,
    occlusion: torch.Tensor,
    regression: torch.Tensor | None,
    confidence_threshold: float,
    maximum: int,
) -> torch.Tensor:
    """Decodifica ``[x, y, radio, oclusión, score, clase]``."""
    batch, categories, height, width = heatmap.size()
    if batch != 1 or categories != 1:
        raise ValueError("El checkpoint sólo soporta batch=1 y una clase")
    if maximum > height * width:
        raise ValueError(
            f"MAX_DETECTIONS={maximum} excede el mapa de {height * width} puntos"
        )

    scores, initial_indices, _, _, _ = _topk(heatmap, maximum)
    selected_occlusion = _select_occlusion_candidates(
        scores,
        initial_indices,
        occlusion,
        confidence_threshold,
    )

    occlusion_scores, indices, classes, ys, xs = _topk(
        _nms(selected_occlusion), maximum
    )
    if regression is not None:
        offsets = _transpose_and_gather(regression, indices).view(
            batch, maximum, 2
        )
        xs = xs.view(batch, maximum, 1) + offsets[:, :, 0:1]
        ys = ys.view(batch, maximum, 1) + offsets[:, :, 1:2]
    else:
        xs = xs.view(batch, maximum, 1) + 0.5
        ys = ys.view(batch, maximum, 1) + 0.5

    radii = _transpose_and_gather(radius, indices).view(batch, maximum, 1)
    occlusions = _transpose_and_gather(occlusion, indices).view(
        batch, maximum, 1
    )
    classes = classes.view(batch, maximum, 1).float()
    occlusion_scores = occlusion_scores.view(batch, maximum, 1)
    circles = torch.cat([xs, ys, radii], dim=2)
    return torch.cat(
        [circles, occlusions, occlusion_scores, classes], dim=2
    )
