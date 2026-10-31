from numbers import Real

import numpy as np

from ..core.validation import _validate_vectors


class VectorStore:
    """Contiguous in-memory vector storage for a single collection."""

    def __init__(
        self,
        dimension: int,
        initial_capacity: int = 16,
        dtype: np.dtype = np.float32,
    ) -> None:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(
            dimension, (int, np.integer)
        ):
            raise TypeError("dimension must be an integer")
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        if isinstance(initial_capacity, (bool, np.bool_)) or not isinstance(
            initial_capacity, (int, np.integer)
        ):
            raise TypeError("initial_capacity must be an integer")
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")

        self._dimension = int(dimension)
        self._capacity = int(initial_capacity)
        self._count = 0
        self._vectors = np.empty(
            (self._capacity, self._dimension), dtype=np.dtype(dtype)
        )
        self._id_to_slot: dict[str, int] = {}
        self._slot_to_id: list[str] = []

    def _assert_invariants(self) -> None:
        assert 0 <= self._count <= self._capacity
        assert self._vectors.shape == (self._capacity, self._dimension)
        assert len(self._id_to_slot) == self._count
        assert len(self._slot_to_id) == self._count
        assert set(self._id_to_slot.values()) == set(range(self._count))

        for record_id, slot in self._id_to_slot.items():
            assert 0 <= slot < self._count
            assert self._slot_to_id[slot] == record_id

    def _ensure_capacity(self, required_capacity: int) -> None:
        if isinstance(required_capacity, (bool, np.bool_)) or not isinstance(
            required_capacity, (int, np.integer)
        ):
            raise TypeError("required_capacity must be an integer")
        if required_capacity < 0:
            raise ValueError("required_capacity must be nonnegative")
        if required_capacity <= self._capacity:
            return

        new_capacity = self._capacity
        while new_capacity < required_capacity:
            new_capacity *= 2

        new_vectors = np.empty(
            (new_capacity, self._dimension), dtype=self._vectors.dtype
        )
        new_vectors[: self._count] = self._vectors[: self._count]

        self._vectors = new_vectors
        self._capacity = new_capacity

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        vector_array = np.asarray(vector)
        if vector_array.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        if vector_array.size == 0:
            raise ValueError("vector must contain at least one component")
        if vector_array.dtype.kind not in "iuf" and not (
            vector_array.dtype.kind == "O"
            and all(
                isinstance(value, Real) and not isinstance(value, bool)
                for value in vector_array
            )
        ):
            raise TypeError("vector components must be real numeric values")

        validated_vector = _validate_vectors(vector_array[np.newaxis, :])[0]
        with np.errstate(over="ignore", invalid="ignore"):
            stored_vector = validated_vector.astype(np.float32)
        if not np.isfinite(stored_vector).all():
            raise ValueError("vector components must be finite in float32 storage")
        if stored_vector.size != self._dimension:
            raise ValueError(f"vector dimension must be {self._dimension}")
        return stored_vector

    def add(self, record_id: str, vector: np.ndarray) -> int:
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")
        if not record_id.strip():
            raise ValueError("record_id must be nonempty")
        if record_id in self._id_to_slot:
            raise ValueError(f"record ID already exists: {record_id}")

        stored_vector = self._prepare_vector(vector)
        slot = self._count
        self._ensure_capacity(slot + 1)
        self._vectors[slot] = stored_vector
        self._slot_to_id.append(record_id)
        try:
            self._id_to_slot[record_id] = slot
        except BaseException:
            self._slot_to_id.pop()
            raise
        self._count += 1
        return slot

    def update(self, record_id: str, vector: np.ndarray) -> None:
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")
        if not record_id.strip():
            raise ValueError("record_id must be nonempty")
        try:
            slot = self._id_to_slot[record_id]
        except KeyError:
            raise KeyError(f"record ID does not exist: {record_id}") from None

        stored_vector = self._prepare_vector(vector)
        self._vectors[slot] = stored_vector

    def delete(self, record_id: str) -> None:
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")
        try:
            slot = self._id_to_slot[record_id]
        except KeyError:
            raise KeyError(f"record ID does not exist: {record_id}") from None

        last_slot = self._count - 1
        if slot != last_slot:
            moved_id = self._slot_to_id[last_slot]
            self._vectors[slot] = self._vectors[last_slot]
            self._slot_to_id[slot] = moved_id
            self._id_to_slot[moved_id] = slot

        del self._id_to_slot[record_id]
        self._slot_to_id.pop()
        self._count -= 1

    def get(self, record_id: str) -> np.ndarray:
        try:
            slot = self._id_to_slot[record_id]
        except KeyError:
            raise KeyError(f"record ID does not exist: {record_id}") from None
        return self._vectors[slot].copy()

    def get_batch(self, record_ids: list[str]) -> np.ndarray:
        """Return vectors in requested order; repeated IDs are allowed.

        Raises KeyError if any ID is missing. An empty request returns an
        empty matrix with shape (0, dimension).
        """
        slots = []
        for record_id in record_ids:
            try:
                slots.append(self._id_to_slot[record_id])
            except KeyError:
                raise KeyError(f"record ID does not exist: {record_id}") from None
        return self._vectors[slots].copy()

    def get_all_active(self) -> tuple[list[str], np.ndarray]:
        """Return active IDs and vectors in matching slot order."""
        return self._slot_to_id.copy(), self._vectors[: self._count].copy()

    def contains(self, record_id: str) -> bool:
        return record_id in self._id_to_slot

    def count(self) -> int:
        return self._count
