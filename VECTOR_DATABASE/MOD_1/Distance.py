import numpy as np
from collections.abc import Callable
from typing import Any, Tuple

def normalize_vectors(vectors:np.ndarray) -> np.ndarray:

        vectors=np.asarray(vectors,dtype=np.float64)

        if vectors.ndim != 2:
            raise ValueError("Expected 2d Matrix of Vectors")

        if not np.isfinite(vectors).all():
            raise ValueError("Vector Contains Nan or Infinity")

        max_values = np.max(np.abs(vectors), axis=1, keepdims=True)

        if np.any(max_values == 0):
             raise ValueError("Cannot Normalize a zero Vector")

        scaled_vectors = vectors / max_values

        norms = np.linalg.norm(scaled_vectors, axis=1, keepdims=True)

        if not np.isfinite(norms).all():
            raise ValueError("Vector norm is not finite")

        return scaled_vectors / norms

def euclidean_distance(vectors1:np.ndarray, vectors2:np.ndarray,flag:bool=False) -> np.ndarray:

       vectors1 =np.asarray(vectors1, dtype=np.float64)
       vectors2 =np.asarray(vectors2, dtype=np.float64)

       if vectors1.ndim !=2 or vectors2.ndim !=2:
             raise ValueError("Expected 2d Matrix of Vectors")

       if not np.isfinite(vectors1).all() or not np.isfinite(vectors2).all():
                raise ValueError("Vector Contains Nan or Infinity")
       
       if vectors1.shape[1] != vectors2.shape[1]:
                raise ValueError("Vectors must have the same dimension")

       if flag:
            diff = vectors1[:, np.newaxis, :] - vectors2[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=-1))
       else:
             
        m1,n1=vectors1.shape
        m2,_=vectors2.shape
        distances=np.zeros((m1,m2),dtype=np.float64)
        for i in range(m1):
            for j in range(m2):
                total =0.0
                for count in range(n1):
                    total += np.square(vectors1[i][count] - vectors2[j][count])
                distances[i][j] = np.sqrt(total)
        return distances  


def sqeuclidean_distance(vectors1:np.ndarray, vectors2:np.ndarray,flag:bool=False) -> np.ndarray:

       vectors1 =np.asarray(vectors1, dtype=np.float64)
       vectors2 =np.asarray(vectors2, dtype=np.float64)

       if vectors1.ndim !=2 or vectors2.ndim !=2:
             raise ValueError("Expected 2d Matrix of Vectors")

       if not np.isfinite(vectors1).all() or not np.isfinite(vectors2).all():
                raise ValueError("Vector Contains Nan or Infinity")
       
       if vectors1.shape[1] != vectors2.shape[1]:
                raise ValueError("Vectors must have the same dimension")

       if flag:
            diff = vectors1[:, np.newaxis, :] - vectors2[np.newaxis, :, :]
            return (np.sum(diff ** 2, axis=-1))
       else:
             
        m1,n1=vectors1.shape
        m2,_=vectors2.shape
        distances=np.zeros((m1,m2),dtype=np.float64)
        for i in range(m1):
            for j in range(m2):
                total =0.0
                for count in range(n1):
                    total += np.square(vectors1[i][count] - vectors2[j][count])
                distances[i][j] = total
        return distances  

def cosine_similarity(vectors1:np.ndarray, vectors2:np.ndarray) -> np.ndarray:

    vectors1 = np.asarray(vectors1, dtype=np.float64)
    vectors2 = np.asarray(vectors2, dtype=np.float64)

    if vectors1.ndim != 2 or vectors2.ndim != 2:
        raise ValueError("Expected each input to be a 2d matrix of vectors")

    if vectors1.shape[1] != vectors2.shape[1]:
        raise ValueError("Vectors must have the same dimension")

    if not np.isfinite(vectors1).all() or not np.isfinite(vectors2).all():
        raise ValueError("Vector Contains Nan or Infinity")

    normalized1 = np.zeros_like(vectors1)
    normalized2 = np.zeros_like(vectors2)
    nonzero_rows1 = np.any(vectors1 != 0, axis=1)
    nonzero_rows2 = np.any(vectors2 != 0, axis=1)

    if np.any(nonzero_rows1):
        normalized1[nonzero_rows1] = normalize_vectors(vectors1[nonzero_rows1])

    if np.any(nonzero_rows2):
        normalized2[nonzero_rows2] = normalize_vectors(vectors2[nonzero_rows2])

    return normalized1 @ normalized2.T

def cosine_distance(vectors1:np.ndarray, vectors2:np.ndarray) -> np.ndarray:
    return 1 - cosine_similarity(vectors1, vectors2)        

def inner_product(vectors1:np.ndarray, vectors2:np.ndarray) -> np.ndarray:    
    
    vectors1 = np.asarray(vectors1, dtype=np.float64)
    vectors2 = np.asarray(vectors2, dtype=np.float64)

    if vectors1.ndim != 2 or vectors2.ndim != 2:
        raise ValueError("Expected each input to be a 2d matrix of vectors")

    if vectors1.shape[1] != vectors2.shape[1]:
        raise ValueError("Vectors must have the same dimension")

    if not np.isfinite(vectors1).all() or not np.isfinite(vectors2).all():
        raise ValueError("Vector Contains Nan or Infinity")

    return vectors1 @ vectors2.T

def k_nearest_neighbors(vectors1:np.ndarray, vectors2:np.ndarray, k:int,kind:str = 'euclidean') -> Tuple[np.ndarray, np.ndarray]:

    vectors1 = np.asarray(vectors1, dtype=np.float64)

    vectors2 = np.asarray(vectors2, dtype=np.float64)

    if vectors2.ndim != 2:
        raise ValueError("vectors2 must be a 2D matrix")

    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if not 1 <= k <= vectors2.shape[0]:
        raise ValueError("k must be between 1 and the number of candidate vectors")

    if kind == "euclidean":
        scores = euclidean_distance(vectors1, vectors2)
        largest = False

    elif kind == "sqeuclidean":
        scores = sqeuclidean_distance(vectors1, vectors2)
        largest = False

    elif kind == "cosine":
        scores = cosine_distance(vectors1, vectors2)
        largest = False

    elif kind == "inner_product":
        scores = inner_product(vectors1, vectors2)
        largest = True

    else:
        raise ValueError("Unsupported distance type")

    if largest:
        scores = -scores

    indices = np.argpartition(scores, kth=k-1, axis=1)[:, :k]

    selected_scores = np.take_along_axis(scores, indices, axis=1)

    sorted_positions = np.argsort(selected_scores, axis=1)

    return np.take_along_axis(indices, sorted_positions, axis=1), np.take_along_axis(selected_scores, sorted_positions, axis=1) 


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    return int(batch_size)


def normalize_vectors_batch(vectors: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("Expected 2d Matrix of Vectors")
    batch_size = _validate_batch_size(batch_size)

    if vectors.shape[0] == 0:
        return normalize_vectors(vectors)

    return np.concatenate(
        [normalize_vectors(vectors[start:start + batch_size])
         for start in range(0, vectors.shape[0], batch_size)],
        axis=0,
    )


def _pairwise_in_batches(
    function: Callable[..., np.ndarray],
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    batch_size: int,
    **kwargs: Any,
) -> np.ndarray:
    vectors1 = np.asarray(vectors1, dtype=np.float64)
    vectors2 = np.asarray(vectors2, dtype=np.float64)
    if vectors1.ndim != 2 or vectors2.ndim != 2:
        raise ValueError("Expected 2d Matrix of Vectors")
    batch_size = _validate_batch_size(batch_size)

    if vectors1.shape[0] == 0 or vectors2.shape[0] == 0:
        return function(vectors1, vectors2, **kwargs)

    result = np.empty((vectors1.shape[0], vectors2.shape[0]), dtype=np.float64)
    for start1 in range(0, vectors1.shape[0], batch_size):
        end1 = start1 + batch_size
        for start2 in range(0, vectors2.shape[0], batch_size):
            end2 = start2 + batch_size
            result[start1:end1, start2:end2] = function(
                vectors1[start1:end1],
                vectors2[start2:end2],
                **kwargs,
            )
    return result


def euclidean_distance_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    flag: bool = False,
    batch_size: int = 1024,
) -> np.ndarray:
    return _pairwise_in_batches(
        euclidean_distance, vectors1, vectors2, batch_size, flag=flag
    )


def sqeuclidean_distance_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    flag: bool = False,
    batch_size: int = 1024,
) -> np.ndarray:
    return _pairwise_in_batches(
        sqeuclidean_distance, vectors1, vectors2, batch_size, flag=flag
    )


def cosine_similarity_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    return _pairwise_in_batches(
        cosine_similarity, vectors1, vectors2, batch_size
    )


def cosine_distance_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    return _pairwise_in_batches(
        cosine_distance, vectors1, vectors2, batch_size
    )


def inner_product_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    return _pairwise_in_batches(
        inner_product, vectors1, vectors2, batch_size
    )


def k_nearest_neighbors_batch(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    k: int,
    kind: str = "euclidean",
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    vectors1 = np.asarray(vectors1, dtype=np.float64)
    if vectors1.ndim != 2:
        raise ValueError("vectors1 must be a 2D matrix")
    batch_size = _validate_batch_size(batch_size)

    if vectors1.shape[0] == 0:
        return k_nearest_neighbors(vectors1, vectors2, k, kind)

    batches = [
        k_nearest_neighbors(vectors1[start:start + batch_size], vectors2, k, kind)
        for start in range(0, vectors1.shape[0], batch_size)
    ]
    return (
        np.concatenate([batch[0] for batch in batches], axis=0),
        np.concatenate([batch[1] for batch in batches], axis=0),
    )
