import unittest

import numpy as np

from VECTOR_DATABASE.vectordb.core.collection import Collection
from VECTOR_DATABASE.vectordb.core.models import CollectionConfig, VectorRecord


class CollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collection = Collection(CollectionConfig("documents", 3))

    def test_insert_and_retrieve_record(self) -> None:
        record = VectorRecord(
            "doc_001",
            np.array([0.1, 0.2, 0.7]),
            {"category": "AI"},
        )

        self.collection.insert(record)

        self.assertTrue(self.collection.contains("doc_001"))
        self.assertEqual(self.collection.count(), 1)
        retrieved = self.collection.get("doc_001")
        self.assertIsNotNone(retrieved)
        assert retrieved is not None
        self.assertEqual(retrieved.id, "doc_001")
        np.testing.assert_array_equal(retrieved.vector, record.vector)
        self.assertEqual(retrieved.metadata, {"category": "AI"})
        self.assertEqual(self.collection.list_ids(), ["doc_001"])
        self.assertIsNone(self.collection.get("missing"))

    def test_rejects_duplicate_ids_and_wrong_dimensions(self) -> None:
        record = VectorRecord("doc_001", np.array([1.0, 2.0, 3.0]), {})
        self.collection.insert(record)

        with self.assertRaisesRegex(ValueError, "already exists"):
            self.collection.insert(record)
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.collection.insert(VectorRecord("doc_002", np.array([1.0]), {}))
        self.assertEqual(self.collection.count(), 1)

    def test_inserted_and_retrieved_records_do_not_expose_stored_mutable_data(self) -> None:
        vector = np.array([1.0, 2.0, 3.0])
        metadata = {"tags": ["original"]}
        record = VectorRecord("doc_001", vector, metadata)

        self.collection.insert(record)
        vector[0] = 9.0
        metadata["tags"].append("changed")

        retrieved = self.collection.get("doc_001")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved.vector, [1.0, 2.0, 3.0])
        self.assertEqual(retrieved.metadata, {"tags": ["original"]})

        retrieved.vector[0] = 8.0
        retrieved.metadata["tags"].append("also changed")
        stored_again = self.collection.get("doc_001")
        assert stored_again is not None
        np.testing.assert_array_equal(stored_again.vector, [1.0, 2.0, 3.0])
        self.assertEqual(stored_again.metadata, {"tags": ["original"]})

    def test_rejects_zero_vectors_only_for_cosine(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero vectors"):
            self.collection.insert(VectorRecord("zero", np.zeros(3), {}))

        for metric in ("euclidean", "sqeuclidean", "inner_product"):
            with self.subTest(metric=metric):
                collection = Collection(CollectionConfig("vectors", 3, metric=metric))
                collection.insert(VectorRecord("zero", np.zeros(3), {}))
                self.assertEqual(collection.count(), 1)

    def test_rejects_non_records_and_invalid_configurations(self) -> None:
        with self.assertRaises(TypeError):
            Collection(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.collection.insert(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
