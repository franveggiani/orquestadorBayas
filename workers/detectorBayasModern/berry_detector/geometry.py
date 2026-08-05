"""Transformaciones geométricas usadas antes y después de la red."""

from __future__ import annotations

import cv2
import numpy as np


def _third_point(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    direction = first - second
    return second + np.array([-direction[1], direction[0]], dtype=np.float32)


def affine_transform(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float32)
    return np.dot(transform, homogeneous)[:2]


def get_affine_transform(
    center: np.ndarray,
    scale: float | np.ndarray | list[float],
    rotation: float,
    output_size: tuple[int, int] | list[int],
    inverse: bool = False,
) -> np.ndarray:
    if not isinstance(scale, (np.ndarray, list)):
        scale_array = np.array([scale, scale], dtype=np.float32)
    else:
        scale_array = np.asarray(scale, dtype=np.float32)

    source_width = scale_array[0]
    destination_width, destination_height = output_size
    radians = np.pi * rotation / 180
    source_direction = np.array(
        [source_width * 0.5 * np.sin(radians),
         -source_width * 0.5 * np.cos(radians)],
        dtype=np.float32,
    )
    destination_direction = np.array(
        [0, destination_width * -0.5], dtype=np.float32
    )

    source = np.zeros((3, 2), dtype=np.float32)
    destination = np.zeros((3, 2), dtype=np.float32)
    source[0] = center
    source[1] = center + source_direction
    destination[0] = [destination_width * 0.5, destination_height * 0.5]
    destination[1] = destination[0] + destination_direction
    source[2] = _third_point(source[0], source[1])
    destination[2] = _third_point(destination[0], destination[1])

    if inverse:
        return cv2.getAffineTransform(destination, source)
    return cv2.getAffineTransform(source, destination)


def transform_predictions(
    coordinates: np.ndarray,
    center: np.ndarray,
    scale: float | np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    transformed = np.zeros(coordinates.shape, dtype=np.float32)
    inverse = get_affine_transform(
        center, scale, 0, output_size, inverse=True
    )
    for index, coordinate in enumerate(coordinates):
        transformed[index, :2] = affine_transform(coordinate[:2], inverse)
    return transformed
