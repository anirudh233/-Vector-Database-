import numpy as np

def normalize_vectors(vectors:np.ndarray) -> np.ndarray:

        vectors=np.asarray(vectors,dtype=np.float32)

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


