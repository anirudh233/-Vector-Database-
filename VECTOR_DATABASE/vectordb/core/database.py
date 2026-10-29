from .collection import Collection
from .models import CollectionConfig


class VectorDatabase:
    """Registry for independently configured vector collections."""

    def __init__(self) -> None:
        self._collections: dict[str, Collection] = {}

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index_type: str = "flat",
    ) -> Collection:
        config = CollectionConfig(
            name=name,
            dimension=dimension,
            metric=metric,
            index_type=index_type,
        )
        if config.name in self._collections:
            raise ValueError(f"collection already exists: {config.name}")

        collection = Collection(config)
        self._collections[config.name] = collection
        return collection

    def get_collection(self, name: str) -> Collection:
        try:
            return self._collections[name]
        except KeyError:
            raise KeyError(f"collection does not exist: {name}") from None

    def list_collections(self) -> list[str]:
        return list(self._collections)

    def delete_collection(self, name: str) -> None:
        try:
            del self._collections[name]
        except KeyError:
            raise KeyError(f"collection does not exist: {name}") from None

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    def count_collection(self) -> int:
            return len(self._collections)
