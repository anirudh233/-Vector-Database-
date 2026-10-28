import numpy as np

from models import CollectionConfig, VectorRecord


class Collection:
    """In-memory collection of vector records governed by one configuration."""

    def __init__(self, config: CollectionConfig) -> None:
        if not isinstance(config, CollectionConfig):
            raise TypeError("config must be a CollectionConfig")
        self.config = config
        self._records: dict[str, VectorRecord] = {}

    def insert(self, record: VectorRecord) -> None:
        if not isinstance(record, VectorRecord):
            raise TypeError("record must be a VectorRecord")
        if record.id in self._records:
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

        self._records[validated_record.id] = validated_record

    def get(self, record_id: str) -> VectorRecord | None:
        record = self._records.get(record_id)
        if record is None:
            return None
        return VectorRecord(record.id, record.vector, record.metadata)

    def contains(self, record_id: str) -> bool:
        return record_id in self._records

    def count(self) -> int:
        return len(self._records)

    def list_ids(self) -> list[str]:
        return list(self._records)
