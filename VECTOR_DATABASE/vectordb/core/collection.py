import numpy as np

from ..storage.vector_store import VectorStore
from .models import CollectionConfig, VectorRecord


class Collection:
    """In-memory collection of vector records governed by one configuration."""

    def __init__(self, config: CollectionConfig) -> None:
        if not isinstance(config, CollectionConfig):
            raise TypeError("config must be a CollectionConfig")
        self.config = config
        self._vector_store = VectorStore(dimension=config.dimension)
        self._metadata: dict[str, dict[str, object]] = {}

    def insert(self, record: VectorRecord) -> None:
        if not isinstance(record, VectorRecord):
            raise TypeError("record must be a VectorRecord")
        if self._vector_store.contains(record.id):
            raise ValueError(f"record ID already exists: {record.id}")

        validated_record = VectorRecord(
            id=record.id,
            vector=record.vector,
            metadata=record.metadata,
        )

        if validated_record.vector.size != self.config.dimension:
            raise ValueError(
                f"record vector dimension must be {self.config.dimension}"
            )
        if self.config.metric == "cosine" and np.all(validated_record.vector == 0):
            raise ValueError("cosine collections cannot contain zero vectors")

        self._vector_store.add(validated_record.id, validated_record.vector)
        try:
            self._metadata[validated_record.id] = validated_record.metadata
        except BaseException:
            self._metadata.pop(validated_record.id, None)
            self._vector_store.delete(validated_record.id)
            raise

    def get(self, record_id: str) -> VectorRecord | None:
        if not self._vector_store.contains(record_id):
            return None
        return VectorRecord(
            record_id,
            self._vector_store.get(record_id),
            self._metadata[record_id],
        )

    def contains(self, record_id: str) -> bool:
        return self._vector_store.contains(record_id)

    def count(self) -> int:
        return self._vector_store.count()

    def list_ids(self) -> list[str]:
        return list(self._metadata)
