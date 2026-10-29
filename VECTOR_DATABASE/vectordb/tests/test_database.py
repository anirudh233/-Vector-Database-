import unittest

import numpy as np

from VECTOR_DATABASE.vectordb.core.database import VectorDatabase
from VECTOR_DATABASE.vectordb.core.models import VectorRecord


class VectorDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = VectorDatabase()

    def test_create_and_list_collections(self) -> None:
        documents = self.database.create_collection("documents", 3)
        images = self.database.create_collection(
            "images", 2, metric="euclidean", index_type="flat"
        )

        self.assertIs(self.database.get_collection("documents"), documents)
        self.assertIs(self.database.get_collection("images"), images)
        self.assertEqual(self.database.list_collections(), ["documents", "images"])
        self.assertTrue(self.database.has_collection("documents"))
        self.assertTrue(self.database.has_collection("images"))

    def test_rejects_duplicate_collection_names(self) -> None:
        self.database.create_collection("documents", 3)

        with self.assertRaisesRegex(ValueError, "already exists"):
            self.database.create_collection("documents", 4)

        self.assertEqual(self.database.list_collections(), ["documents"])
        self.assertEqual(self.database.get_collection("documents").config.dimension, 3)

    def test_missing_collection_lookup_and_delete_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(KeyError, "collection does not exist: missing"):
            self.database.get_collection("missing")
        with self.assertRaisesRegex(KeyError, "collection does not exist: missing"):
            self.database.delete_collection("missing")

    def test_delete_removes_collection_from_registry(self) -> None:
        self.database.create_collection("documents", 3)

        self.database.delete_collection("documents")

        self.assertFalse(self.database.has_collection("documents"))
        self.assertEqual(self.database.list_collections(), [])
        with self.assertRaisesRegex(KeyError, "collection does not exist: documents"):
            self.database.get_collection("documents")

    def test_collections_keep_records_independent(self) -> None:
        documents = self.database.create_collection("documents", 2)
        images = self.database.create_collection("images", 2)

        documents.insert(VectorRecord("shared-id", np.array([1.0, 0.0]), {}))

        self.assertTrue(documents.contains("shared-id"))
        self.assertFalse(images.contains("shared-id"))
        self.assertEqual(documents.count(), 1)
        self.assertEqual(images.count(), 0)


if __name__ == "__main__":
    unittest.main()
