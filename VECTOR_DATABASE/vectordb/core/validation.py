import numpy as np
from typing import Any, Tuple


def _validate_vectors(vectors: np.ndarray, name: str = "vectors") -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if vectors.shape[1] == 0:
        raise ValueError("Vectors must have at least one dimension")
    if not np.isfinite(vectors).all():
        raise ValueError("Vector Contains Nan or Infinity")
    return vectors


def _validate_pair(
    vectors1: np.ndarray, vectors2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    vectors1 = _validate_vectors(vectors1, "vectors1")
    vectors2 = _validate_vectors(vectors2, "vectors2")
    if vectors1.shape[1] != vectors2.shape[1]:
        raise ValueError("Vectors must have the same dimension")
    return vectors1, vectors2


def _validate_batch_size(batch_size: int, name: str = "batch_size") -> int:
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    if batch_size <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return int(batch_size)


def _validate_k(k: int, candidate_count: int) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if not 1 <= k <= candidate_count:
        raise ValueError("k must be between 1 and the number of candidate vectors")
    return int(k)