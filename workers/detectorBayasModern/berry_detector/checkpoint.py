"""Carga estricta y segura de checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


class CheckpointError(RuntimeError):
    """El checkpoint no puede utilizarse con la arquitectura actual."""


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise CheckpointError("El checkpoint no contiene un mapping de pesos")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise CheckpointError("'state_dict' no es un mapping")
    if not all(isinstance(key, str) for key in state_dict):
        raise CheckpointError("Las claves del state_dict deben ser strings")
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise CheckpointError("El state_dict contiene valores que no son tensores")
    return state_dict


def _remove_data_parallel_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_checkpoint(model: nn.Module, model_path: Path) -> int | None:
    """Carga ``model_path`` y devuelve la época guardada, si existe."""
    if not model_path.is_file():
        raise CheckpointError(f"No existe el checkpoint: {model_path}")
    try:
        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise CheckpointError(
            f"No se pudo leer de forma segura el checkpoint {model_path}: {exc}"
        ) from exc

    state_dict = _remove_data_parallel_prefix(_extract_state_dict(checkpoint))
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatched = sorted(
        key
        for key in expected.keys() & state_dict.keys()
        if expected[key].shape != state_dict[key].shape
    )
    problems: list[str] = []
    if missing:
        problems.append(f"faltan claves: {missing}")
    if unexpected:
        problems.append(f"sobran claves: {unexpected}")
    if mismatched:
        details = [
            f"{key}: esperado {tuple(expected[key].shape)}, "
            f"recibido {tuple(state_dict[key].shape)}"
            for key in mismatched
        ]
        problems.append(f"shapes incompatibles: {details}")
    if problems:
        raise CheckpointError("; ".join(problems))

    model.load_state_dict(state_dict, strict=True)
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None
    return int(epoch) if epoch is not None else None
