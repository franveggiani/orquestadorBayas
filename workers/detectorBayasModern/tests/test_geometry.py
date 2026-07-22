import numpy as np

from berry_detector.geometry import get_affine_transform, transform_predictions


def test_center_is_preserved_by_square_affine_transform() -> None:
    center = np.array([50.0, 50.0], dtype=np.float32)
    transform = get_affine_transform(center, 100.0, 0, (100, 100))

    transformed = transform @ np.array([50.0, 50.0, 1.0])

    assert transformed.tolist() == [50.0, 50.0]


def test_predictions_return_to_original_coordinates() -> None:
    center = np.array([50.0, 50.0], dtype=np.float32)
    coordinates = np.array([[50.0, 50.0]], dtype=np.float32)

    result = transform_predictions(coordinates, center, 100.0, (100, 100))

    np.testing.assert_allclose(result, coordinates)
