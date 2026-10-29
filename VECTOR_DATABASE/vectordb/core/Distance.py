import numpy as np
from collections.abc import Callable
from typing import Any, Tuple
from .validation import (
    _validate_batch_size,
    _validate_k,
    _validate_pair,
    _validate_vectors,
)

def _normalize_allow_zero(vectors: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(vectors)
    nonzero_rows = np.any(vectors != 0, axis=1)
    if np.any(nonzero_rows):
        normalized[nonzero_rows] = _normalize_nonzero_rows(vectors[nonzero_rows])
    return normalized


def _normalize_nonzero_rows(vectors: np.ndarray) -> np.ndarray:
    scaled_vectors = vectors / np.max(np.abs(vectors), axis=1, keepdims=True)
    norms = np.linalg.norm(scaled_vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("Vector norm is not finite")
    return scaled_vectors / norms


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = _validate_vectors(vectors)
    if np.any(np.all(vectors == 0, axis=1)):
        raise ValueError("Cannot Normalize a zero Vector")
    return _normalize_nonzero_rows(vectors)


def _euclidean_kernel(
    vectors1: np.ndarray, vectors2: np.ndarray, vectorized: bool
) -> np.ndarray:
    if vectorized:
        diff = vectors1[:, np.newaxis, :] - vectors2[np.newaxis, :, :]
        np.square(diff, out=diff)
        squared_distances = np.sum(diff, axis=-1)
        return np.sqrt(squared_distances)
    
    distances = np.empty((vectors1.shape[0], vectors2.shape[0]), dtype=np.float64)
    
    for i in range(vectors1.shape[0]):
        for j in range(vectors2.shape[0]):
            diff = vectors1[i] - vectors2[j]
            distances[i, j] = np.sqrt(np.sum(diff * diff))
    return distances


def _sqeuclidean_kernel(
    vectors1: np.ndarray, vectors2: np.ndarray, vectorized: bool
) -> np.ndarray:
    if vectorized:
        diff = vectors1[:, np.newaxis, :] - vectors2[np.newaxis, :, :]
        np.square(diff, out=diff)   
        squared_distances = np.sum(diff, axis=-1)
        return squared_distances

    distances = np.empty((vectors1.shape[0], vectors2.shape[0]), dtype=np.float64)
    for i in range(vectors1.shape[0]):
        for j in range(vectors2.shape[0]):
            diff = vectors1[i] - vectors2[j]
            distances[i, j] = np.sum(diff * diff)
    return distances


def euclidean_distance(
    vectors1: np.ndarray, vectors2: np.ndarray, vectorized: bool = True
) -> np.ndarray:
    vectors1, vectors2 = _validate_pair(vectors1, vectors2)
    return _euclidean_kernel(vectors1, vectors2, vectorized)


def sqeuclidean_distance(
    vectors1: np.ndarray, vectors2: np.ndarray, vectorized: bool = True
) -> np.ndarray:
    vectors1, vectors2 = _validate_pair(vectors1, vectors2)
    return _sqeuclidean_kernel(vectors1, vectors2, vectorized)


def cosine_similarity(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
    vectors1, vectors2 = _validate_pair(vectors1, vectors2)
    return _normalize_allow_zero(vectors1) @ _normalize_allow_zero(vectors2).T


def cosine_distance(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
    return 1 - cosine_similarity(vectors1, vectors2)


def inner_product(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
    vectors1, vectors2 = _validate_pair(vectors1, vectors2)
    return vectors1 @ vectors2.T


def _metric_scores(
    vectors1: np.ndarray, vectors2: np.ndarray, kind: str, vectorized: bool = True
) -> Tuple[np.ndarray, bool]:
    if kind == "euclidean":
        return _euclidean_kernel(vectors1, vectors2, vectorized), False
    if kind == "sqeuclidean":
        return _sqeuclidean_kernel(vectors1, vectors2, vectorized), False
    if kind == "cosine":
        scores = _normalize_allow_zero(vectors1) @ _normalize_allow_zero(vectors2).T
        return 1 - scores, False
    if kind == "inner_product":
        return vectors1 @ vectors2.T, True
    raise ValueError("Unsupported distance type")

def _select_top_k(
    scores: np.ndarray, k: int, largest: bool
) -> Tuple[np.ndarray, np.ndarray]:
    ranking_scores = -scores if largest else scores
    all_indices = np.broadcast_to(
        np.arange(scores.shape[1], dtype=np.intp), scores.shape
    )
    sorted_indices = np.empty((scores.shape[0], k), dtype=np.intp)
    sorted_scores = np.empty((scores.shape[0], k), dtype=np.float64)
    for row in range(scores.shape[0]):
        row_scores = ranking_scores[row]
        partition = np.argpartition(row_scores, kth=k - 1)[:k]
        threshold = np.max(row_scores[partition])
        better = np.flatnonzero(row_scores < threshold)
        tied = np.flatnonzero(row_scores == threshold)
        selected = np.concatenate((better, tied[:k - better.size]))
        order = np.lexsort((all_indices[row, selected], row_scores[selected]))
        selected = selected[order]
        sorted_indices[row] = selected
        sorted_scores[row] = row_scores[selected]
    return sorted_indices, -sorted_scores if largest else sorted_scores


def k_nearest_neighbors(
    vectors1: np.ndarray, vectors2: np.ndarray, k: int, kind: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    vectors1, vectors2 = _validate_pair(vectors1, vectors2)
    k = _validate_k(k, vectors2.shape[0])
    scores, largest = _metric_scores(vectors1, vectors2, kind)
    return _select_top_k(scores, k, largest)


def normalize_vectors_batch(
    vectors: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    vectors = _validate_vectors(vectors)
    batch_size = _validate_batch_size(batch_size)
    normalized = np.empty_like(vectors)
    for start in range(0, vectors.shape[0], batch_size):
        end = min(start + batch_size, vectors.shape[0])
        normalized[start:end] = normalize_vectors(vectors[start:end])
    return normalized


def _pairwise_in_batches(
    function: Callable[..., np.ndarray],
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    batch_size: int,
    row_batch_size: int | None,
    candidate_batch_size: int | None,
    **kwargs: Any,
) -> np.ndarray:
    vectors1, vectors2 = _validate_pair(vectors1, vectors2)
    default_size = _validate_batch_size(batch_size)
    row_size = _validate_batch_size(
        default_size if row_batch_size is None else row_batch_size,
        "row_batch_size",
    )
    candidate_size = _validate_batch_size(
        default_size if candidate_batch_size is None else candidate_batch_size,
        "candidate_batch_size",
    )

    result = np.empty((vectors1.shape[0], vectors2.shape[0]), dtype=np.float64)
    if function in (cosine_similarity, cosine_distance):
        normalized1 = _normalize_allow_zero(vectors1)
        normalized2 = _normalize_allow_zero(vectors2)
        for row_start in range(0, vectors1.shape[0], row_size):
            row_end = min(row_start + row_size, vectors1.shape[0])
            for candidate_start in range(0, vectors2.shape[0], candidate_size):
                candidate_end = min(candidate_start + candidate_size, vectors2.shape[0])
                similarities = (
                    normalized1[row_start:row_end]
                    @ normalized2[candidate_start:candidate_end].T
                )
                result[row_start:row_end, candidate_start:candidate_end] = (
                    1 - similarities if function is cosine_distance else similarities
                )
        return result

    for row_start in range(0, vectors1.shape[0], row_size):
        row_end = min(row_start + row_size, vectors1.shape[0])
        for candidate_start in range(0, vectors2.shape[0], candidate_size):
            candidate_end = min(candidate_start + candidate_size, vectors2.shape[0])
            left = vectors1[row_start:row_end]
            right = vectors2[candidate_start:candidate_end]
            if function is euclidean_distance:
                block = _euclidean_kernel(left, right, kwargs["vectorized"])
            elif function is sqeuclidean_distance:
                block = _sqeuclidean_kernel(left, right, kwargs["vectorized"])
            else:
                block = left @ right.T
            result[row_start:row_end, candidate_start:candidate_end] = block
    return result


def euclidean_distance_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    vectorized: bool = True,
    batch_size: int = 64,
    row_batch_size: int | None = None,
    candidate_batch_size: int | None = None,
) -> np.ndarray:
    return _pairwise_in_batches(
        euclidean_distance,
        vectors1,
        vectors2,
        batch_size,
        row_batch_size,
        candidate_batch_size,
        vectorized=vectorized,
    )


def sqeuclidean_distance_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    vectorized: bool = True,
    batch_size: int = 64,
    row_batch_size: int | None = None,
    candidate_batch_size: int | None = None,
) -> np.ndarray:
    return _pairwise_in_batches(
        sqeuclidean_distance,
        vectors1,
        vectors2,
        batch_size,
        row_batch_size,
        candidate_batch_size,
        vectorized=vectorized,
    )


def cosine_similarity_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    batch_size: int = 64,
    row_batch_size: int | None = None,
    candidate_batch_size: int | None = None,
) -> np.ndarray:
    return _pairwise_in_batches(
        cosine_similarity, vectors1, vectors2, batch_size, row_batch_size, candidate_batch_size
    )


def cosine_distance_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    batch_size: int = 64,
    row_batch_size: int | None = None,
    candidate_batch_size: int | None = None,
) -> np.ndarray:
    return _pairwise_in_batches(
        cosine_distance, vectors1, vectors2, batch_size, row_batch_size, candidate_batch_size
    )


def inner_product_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    batch_size: int = 64,
    row_batch_size: int | None = None,
    candidate_batch_size: int | None = None,
) -> np.ndarray:
    return _pairwise_in_batches(
        inner_product, vectors1, vectors2, batch_size, row_batch_size, candidate_batch_size
    )


def k_nearest_neighbors_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    k: int,
    kind: str = "euclidean",
    batch_size: int | None = None,
    query_batch_size: int | None = None,
    candidate_batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    vectors1, vectors2 = _validate_pair(vectors1, vectors2)
    k = _validate_k(k, vectors2.shape[0])
    if batch_size is not None:
        if query_batch_size is not None:
            raise ValueError("Specify either batch_size or query_batch_size, not both")
        query_batch_size = batch_size
    query_size = _validate_batch_size(
        32 if query_batch_size is None else query_batch_size, "query_batch_size"
    )
    candidate_size = _validate_batch_size(candidate_batch_size, "candidate_batch_size")

    largest = kind == "inner_product"
    if kind not in ("euclidean", "sqeuclidean", "cosine", "inner_product"):
        raise ValueError("Unsupported distance type")

    normalized_candidates = _normalize_allow_zero(vectors2) if kind == "cosine" else None
    indices_result = np.empty((vectors1.shape[0], k), dtype=np.intp)
    scores_result = np.empty((vectors1.shape[0], k), dtype=np.float64)

    for query_start in range(0, vectors1.shape[0], query_size):
        query_end = min(query_start + query_size, vectors1.shape[0])
        query_vectors = vectors1[query_start:query_end]
        query_normalized = (
            _normalize_allow_zero(query_vectors) if kind == "cosine" else None
        )
        best_indices = np.empty((query_vectors.shape[0], 0), dtype=np.intp)
        best_ranking_scores = np.empty((query_vectors.shape[0], 0), dtype=np.float64)

        for candidate_start in range(0, vectors2.shape[0], candidate_size):
            candidate_end = min(candidate_start + candidate_size, vectors2.shape[0])
            candidate_vectors = vectors2[candidate_start:candidate_end]
            if kind == "cosine":
                similarity = query_normalized @ normalized_candidates[
                    candidate_start:candidate_end
                ].T
                block_scores = 1 - similarity
            else:
                block_scores, _ = _metric_scores(
                    query_vectors, candidate_vectors, kind
                )

            block_ranking_scores = -block_scores if largest else block_scores
            block_indices = np.broadcast_to(
                np.arange(candidate_start, candidate_end, dtype=np.intp),
                block_scores.shape,
            )
            merged_scores = np.concatenate(
                (best_ranking_scores, block_ranking_scores), axis=1
            )
            merged_indices = np.concatenate((best_indices, block_indices), axis=1)
            keep_count = min(k, merged_scores.shape[1])
            selected_scores = np.empty(
                (query_vectors.shape[0], keep_count), dtype=np.float64
            )
            selected_indices = np.empty(
                (query_vectors.shape[0], keep_count), dtype=np.intp
            )
            for row in range(query_vectors.shape[0]):
                order = np.lexsort(
                    (merged_indices[row], merged_scores[row])
                )[:keep_count]
                selected_scores[row] = merged_scores[row, order]
                selected_indices[row] = merged_indices[row, order]
            best_ranking_scores = selected_scores
            best_indices = selected_indices

        indices_result[query_start:query_end] = best_indices
        scores_result[query_start:query_end] = (
            -best_ranking_scores if largest else best_ranking_scores
        )

    return indices_result, scores_result
