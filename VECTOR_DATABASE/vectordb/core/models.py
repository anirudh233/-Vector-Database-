from dataclasses import dataclass
from copy import deepcopy
from numbers import Real
from typing import ClassVar
import numpy as np

from .validation import _validate_vectors



@dataclass(frozen=True)
class CollectionConfig:
    """Immutable collection settings.

    Collections store vectors as float32; numerical operations may promote
    values to float64. The database is responsible for enforcing unique names.
    """

    STORAGE_DTYPE: ClassVar[str] = "float32"
    ALLOWED_METRICS: ClassVar[frozenset[str]] = frozenset(
        {"euclidean", "sqeuclidean", "cosine", "inner_product"}
    )

    name: str
    dimension: int
    metric: str = "cosine"
    index_type: str = "flat"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name.strip():
            raise ValueError("name must be nonempty")

        if isinstance(self.dimension, (bool, np.bool_)) or not isinstance(
        self.dimension, (int, np.integer)
        ):
            raise TypeError("dimension must be an integer")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")

        if not isinstance(self.metric, str):
            raise TypeError("metric must be a string")
        if self.metric not in self.ALLOWED_METRICS:
            raise ValueError(
                f"metric must be one of {', '.join(sorted(self.ALLOWED_METRICS))}"
            )

        if not isinstance(self.index_type, str):
            raise TypeError("index_type must be a string")
        if self.index_type != "flat":
            raise ValueError("index_type must be 'flat'")


def _validate_json_value(value: object, active_containers: set[int]) -> None:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return
    if value_type is float:
        if not np.isfinite(value):
            raise ValueError("metadata float values must be finite")
        return
    if value_type not in (dict, list):
        raise TypeError("metadata values must be JSON-compatible")

    identity = id(value)
    if identity in active_containers:
        raise ValueError("metadata cannot contain circular references")
    active_containers.add(identity)
    try:
        if value_type is dict:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("metadata dictionary keys must be strings")
                _validate_json_value(item, active_containers)
        else:
            for item in value:
                _validate_json_value(item, active_containers)
    finally:
        active_containers.remove(identity)


@dataclass(frozen=True)
class VectorRecord:
    """A record with a stable logical ID, independent of its storage position.

    Collections must reject duplicate IDs on insert. Updating and upserting
    records are intentionally deferred to the collection API.
    """

    id: str
    vector: np.ndarray
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("id must be a string")
        if not self.id.strip():
            raise ValueError("id must be nonempty")

        vector = np.asarray(self.vector)
        if vector.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        if vector.size == 0:
            raise ValueError("vector must contain at least one component")
        if vector.dtype.kind not in "iuf" and not (
            vector.dtype.kind == "O"
            and all(isinstance(value, Real) and not isinstance(value, bool) for value in vector)
        ):
            raise TypeError("vector components must be real numeric values")

        validated_vector = _validate_vectors(vector[np.newaxis, :])[0]
        with np.errstate(over="ignore", invalid="ignore"):
            stored_vector = validated_vector.astype(np.float32)
        if not np.isfinite(stored_vector).all():
            raise ValueError("vector components must be finite in float32 storage")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        _validate_json_value(self.metadata, set())

        object.__setattr__(self, "vector", stored_vector.copy())
        object.__setattr__(self, "metadata", deepcopy(self.metadata))
