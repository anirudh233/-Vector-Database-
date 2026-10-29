import unittest
from dataclasses import FrozenInstanceError
import numpy as np

from VECTOR_DATABASE.vectordb.core.models import CollectionConfig, VectorRecord


class CollectionConfigTests(unittest.TestCase):
    def test_valid_configurations(self) -> None:
        config = CollectionConfig(name="documents", dimension=384)
        self.assertEqual(config.metric, "cosine")
        self.assertEqual(config.index_type, "flat")
        self.assertEqual(config.STORAGE_DTYPE, "float32")

        for metric in CollectionConfig.ALLOWED_METRICS:
            with self.subTest(metric=metric):
                self.assertEqual(
                    CollectionConfig("vectors", 3, metric=metric).metric, metric
                )

    def test_rejects_empty_or_non_string_names(self) -> None:
        for name in ("", "  "):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    CollectionConfig(name, 3)

        with self.assertRaises(TypeError):
            CollectionConfig(123, 3)  # type: ignore[arg-type]

    def test_rejects_invalid_dimensions(self) -> None:
        for dimension in (0, -1):
            with self.subTest(dimension=dimension):
                with self.assertRaises(ValueError):
                    CollectionConfig("vectors", dimension)

        for dimension in (True, 1.5, "3"):
            with self.subTest(dimension=dimension):
                with self.assertRaises(TypeError):
                    CollectionConfig("vectors", dimension)  # type: ignore[arg-type]

    def test_rejects_invalid_metrics_and_index_types(self) -> None:
        for metric in ("manhattan", ""):
            with self.subTest(metric=metric):
                with self.assertRaises(ValueError):
                    CollectionConfig("vectors", 3, metric=metric)

        with self.assertRaises(TypeError):
            CollectionConfig("vectors", 3, metric=1)  # type: ignore[arg-type]

        for index_type in ("ivf", "hnsw", ""):
            with self.subTest(index_type=index_type):
                with self.assertRaises(ValueError):
                    CollectionConfig("vectors", 3, index_type=index_type)

    def test_config_is_immutable(self) -> None:
        config = CollectionConfig("vectors", 3)
        with self.assertRaises(FrozenInstanceError):
            config.dimension = 4  # type: ignore[misc]


class VectorRecordTests(unittest.TestCase):
    def test_accepts_valid_record_and_uses_float32_storage(self) -> None:
        record = VectorRecord(
            id="doc_001",
            vector=np.array([0.12, 0.45, 0.78], dtype=np.float64),
            metadata={"category": "AI", "year": 2026, "active": True, "extra": None},
        )
        self.assertEqual(record.id, "doc_001")
        self.assertEqual(record.vector.dtype, np.dtype(np.float32))
        self.assertEqual(record.vector.shape, (3,))
        self.assertEqual(record.metadata["category"], "AI")

    def test_copies_input_vector_and_nested_metadata(self) -> None:
        vector = np.array([1.0, 2.0], dtype=np.float32)
        metadata = {"tags": ["original"], "details": {"year": 2026}}
        record = VectorRecord("doc_001", vector, metadata)

        vector[0] = 9.0
        metadata["tags"].append("changed")
        metadata["details"]["year"] = 2000

        np.testing.assert_array_equal(record.vector, [1.0, 2.0])
        self.assertEqual(record.metadata, {"tags": ["original"], "details": {"year": 2026}})

    def test_record_identity_is_fixed_and_duplicate_checks_are_not_global(self) -> None:
        first_record = VectorRecord("doc_001", np.array([1.0]), {})
        second_record = VectorRecord("doc_001", np.array([2.0]), {})

        with self.assertRaises(FrozenInstanceError):
            first_record.id = "doc_002"  # type: ignore[misc]
        self.assertEqual(first_record.id, second_record.id)

    def test_rejects_invalid_ids_and_vectors(self) -> None:
        for invalid_id in ("", "  "):
            with self.subTest(id=invalid_id):
                with self.assertRaises(ValueError):
                    VectorRecord(invalid_id, np.array([1.0]), {})

        with self.assertRaises(TypeError):
            VectorRecord(1, np.array([1.0]), {})  # type: ignore[arg-type]

        for vector in (np.array(1.0), np.empty(0), np.array([[1.0]])):
            with self.subTest(vector_shape=vector.shape):
                with self.assertRaises(ValueError):
                    VectorRecord("doc_001", vector, {})

        for vector in (
            np.array([np.nan]),
            np.array([np.inf]),
            np.array(["1"]),
            np.array([True]),
            np.array([1 + 2j]),
            np.array([np.finfo(np.float64).max]),
        ):
            with self.subTest(vector=vector):
                with self.assertRaises((TypeError, ValueError)):
                    VectorRecord("doc_001", vector, {})

    def test_rejects_invalid_or_non_json_metadata(self) -> None:
        for metadata in ([], None):
            with self.subTest(metadata=metadata):
                with self.assertRaises(TypeError):
                    VectorRecord("doc_001", np.array([1.0]), metadata)  # type: ignore[arg-type]

        for metadata in (
            {1: "not a string key"},
            {"value": object()},
            {"value": float("nan")},
            {"value": float("inf")}
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaises((TypeError, ValueError)):
                    VectorRecord("doc_001", np.array([1.0]), metadata)

        circular: list[object] = []
        circular.append(circular)
        with self.assertRaises(ValueError):
            VectorRecord("doc_001", np.array([1.0]), {"circular": circular})


if __name__ == "__main__":
    unittest.main()
