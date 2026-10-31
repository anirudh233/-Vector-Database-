import unittest
from unittest.mock import patch

import numpy as np

from VECTOR_DATABASE.vectordb.storage.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_initializes_empty_contiguous_float32_storage(self) -> None:
        store = VectorStore(dimension=3)

        self.assertEqual(store._dimension, 3)
        self.assertEqual(store._capacity, 16)
        self.assertEqual(store._count, 0)
        self.assertEqual(store._vectors.shape, (16, 3))
        self.assertEqual(store._vectors.dtype, np.dtype(np.float32))
        self.assertTrue(store._vectors.flags.c_contiguous)
        self.assertEqual(store._id_to_slot, {})
        self.assertEqual(store._slot_to_id, [])
        store._assert_invariants()

    def test_accepts_custom_capacity_and_dtype(self) -> None:
        store = VectorStore(2, initial_capacity=5, dtype=np.float64)

        self.assertEqual(store._vectors.shape, (5, 2))
        self.assertEqual(store._vectors.dtype, np.dtype(np.float64))
        store._assert_invariants()

    def test_rejects_invalid_dimensions(self) -> None:
        for dimension in (0, -1):
            with self.subTest(dimension=dimension):
                with self.assertRaises(ValueError):
                    VectorStore(dimension)

        for dimension in (True, 1.5, "3"):
            with self.subTest(dimension=dimension):
                with self.assertRaises(TypeError):
                    VectorStore(dimension)  # type: ignore[arg-type]

    def test_rejects_invalid_capacities(self) -> None:
        for capacity in (0, -1):
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    VectorStore(3, initial_capacity=capacity)

        for capacity in (True, 1.5, "3"):
            with self.subTest(capacity=capacity):
                with self.assertRaises(TypeError):
                    VectorStore(3, initial_capacity=capacity)  # type: ignore[arg-type]

    def test_does_not_allocate_when_capacity_is_sufficient(self) -> None:
        store = VectorStore(2, initial_capacity=4)
        original_vectors = store._vectors

        with patch(
            "VECTOR_DATABASE.vectordb.storage.vector_store.np.empty"
        ) as allocate:
            store._ensure_capacity(4)

        allocate.assert_not_called()
        self.assertIs(store._vectors, original_vectors)
        self.assertEqual(store._capacity, 4)

    def test_expands_geometrically_and_preserves_active_data_and_mappings(self) -> None:
        store = VectorStore(2, initial_capacity=4)
        store._count = 4
        store._vectors[:4] = [[1, 2], [3, 4], [5, 6], [7, 8]]
        store._id_to_slot = {f"id-{slot}": slot for slot in range(4)}
        store._slot_to_id = [f"id-{slot}" for slot in range(4)]
        original_vectors = store._vectors

        store._ensure_capacity(5)

        self.assertEqual(store._capacity, 8)
        self.assertEqual(store._vectors.shape, (8, 2))
        self.assertIsNot(store._vectors, original_vectors)
        np.testing.assert_array_equal(store._vectors[:4], original_vectors[:4])
        self.assertEqual(store._id_to_slot, {f"id-{slot}": slot for slot in range(4)})
        self.assertEqual(store._slot_to_id, [f"id-{slot}" for slot in range(4)])
        self.assertEqual(store._count, 4)
        store._assert_invariants()

    def test_expands_for_large_batch_and_does_not_add_inactive_slots(self) -> None:
        store = VectorStore(2, initial_capacity=4)
        store._count = 2
        store._vectors[:2] = [[1, 2], [3, 4]]
        store._id_to_slot = {"first": 0, "second": 1}
        store._slot_to_id = ["first", "second"]

        store._ensure_capacity(20)

        self.assertEqual(store._capacity, 32)
        self.assertEqual(store._count, 2)
        self.assertEqual(len(store._id_to_slot), 2)
        self.assertEqual(len(store._slot_to_id), 2)
        store._assert_invariants()

    def test_failed_allocation_leaves_store_unchanged(self) -> None:
        store = VectorStore(2, initial_capacity=2)
        store._count = 1
        store._vectors[0] = [1, 2]
        store._id_to_slot = {"first": 0}
        store._slot_to_id = ["first"]
        original_vectors = store._vectors
        original_id_to_slot = store._id_to_slot.copy()
        original_slot_to_id = store._slot_to_id.copy()

        with patch(
            "VECTOR_DATABASE.vectordb.storage.vector_store.np.empty",
            side_effect=MemoryError,
        ):
            with self.assertRaises(MemoryError):
                store._ensure_capacity(3)

        self.assertIs(store._vectors, original_vectors)
        self.assertEqual(store._capacity, 2)
        self.assertEqual(store._count, 1)
        self.assertEqual(store._id_to_slot, original_id_to_slot)
        self.assertEqual(store._slot_to_id, original_slot_to_id)
        np.testing.assert_array_equal(store._vectors[0], [1, 2])
        store._assert_invariants()

    def test_add_inserts_vector_and_registers_id(self) -> None:
        store = VectorStore(2, initial_capacity=2)

        slot = store.add("first", np.array([1.25, 2.5]))

        self.assertEqual(slot, 0)
        self.assertEqual(store._count, 1)
        self.assertEqual(store._id_to_slot, {"first": 0})
        self.assertEqual(store._slot_to_id, ["first"])
        np.testing.assert_array_equal(store._vectors[0], [1.25, 2.5])
        store._assert_invariants()

    def test_add_rejects_invalid_and_duplicate_ids(self) -> None:
        store = VectorStore(2)
        for record_id, error in ((1, TypeError), ("", ValueError), ("  ", ValueError)):
            with self.subTest(record_id=record_id):
                with self.assertRaises(error):
                    store.add(record_id, np.array([1.0, 2.0]))  # type: ignore[arg-type]

        store.add("first", np.array([1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "already exists"):
            store.add("first", np.array([3.0, 4.0]))
        self.assertEqual(store._count, 1)
        store._assert_invariants()

    def test_add_rejects_wrong_dimensions_and_nonfinite_vectors(self) -> None:
        store = VectorStore(2)
        invalid_vectors = (
            (np.array([1.0]), ValueError),
            (np.array([np.nan, 1.0]), ValueError),
            (np.array([np.inf, 1.0]), ValueError),
            (np.array([1e100, 1.0]), ValueError),
            (np.array([1.0 + 2.0j, 2.0]), TypeError),
            (np.array([True, False]), TypeError),
            (np.array(["1", "2"]), TypeError),
        )
        for vector, error in invalid_vectors:
            with self.subTest(vector=vector):
                with self.assertRaises(error):
                    store.add(f"id-{len(store._id_to_slot)}", vector)

        self.assertEqual(store._count, 0)
        self.assertEqual(store._id_to_slot, {})
        self.assertEqual(store._slot_to_id, [])
        store._assert_invariants()

    def test_add_safely_converts_numeric_input_and_copies_it(self) -> None:
        store = VectorStore(2, initial_capacity=2)
        vector = np.array([1.5, 2.25], dtype=np.float64)

        store.add("first", vector)
        vector[0] = 9.0

        self.assertEqual(store._vectors.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(store._vectors[0], [1.5, 2.25])
        store._assert_invariants()

    def test_add_grows_when_full_without_changing_existing_mappings(self) -> None:
        store = VectorStore(2, initial_capacity=2)
        store.add("first", np.array([1.0, 2.0]))
        store.add("second", np.array([3.0, 4.0]))

        slot = store.add("third", np.array([5.0, 6.0]))

        self.assertEqual(slot, 2)
        self.assertEqual(store._capacity, 4)
        self.assertEqual(store._count, 3)
        self.assertEqual(
            store._id_to_slot, {"first": 0, "second": 1, "third": 2}
        )
        self.assertEqual(store._slot_to_id, ["first", "second", "third"])
        np.testing.assert_array_equal(store._vectors[:3], [[1, 2], [3, 4], [5, 6]])
        store._assert_invariants()

    def test_get_returns_copy_for_existing_record(self) -> None:
        store = VectorStore(2)
        store.add("first", np.array([1.5, 2.5]))

        retrieved = store.get("first")
        np.testing.assert_array_equal(retrieved, [1.5, 2.5])
        self.assertTrue(retrieved.flags.owndata)

        retrieved[0] = 100
        np.testing.assert_array_equal(store.get("first"), [1.5, 2.5])

    def test_get_raises_clear_key_error_for_missing_record(self) -> None:
        store = VectorStore(2)

        with self.assertRaisesRegex(KeyError, "record ID does not exist: missing"):
            store.get("missing")

    def test_contains_and_count_follow_insertions(self) -> None:
        store = VectorStore(2)

        self.assertFalse(store.contains("first"))
        self.assertEqual(store.count(), 0)

        store.add("first", np.array([1.0, 2.0]))
        self.assertTrue(store.contains("first"))
        self.assertFalse(store.contains("missing"))
        self.assertEqual(store.count(), 1)

        store.add("second", np.array([3.0, 4.0]))
        self.assertTrue(store.contains("second"))
        self.assertEqual(store.count(), 2)
        store._assert_invariants()

    def test_get_batch_preserves_requested_order_and_allows_repeated_ids(self) -> None:
        store = VectorStore(2)
        store.add("first", np.array([1.0, 2.0]))
        store.add("second", np.array([3.0, 4.0]))
        store.add("third", np.array([5.0, 6.0]))

        vectors = store.get_batch(["third", "first", "third"])

        np.testing.assert_array_equal(
            vectors, [[5.0, 6.0], [1.0, 2.0], [5.0, 6.0]]
        )
        self.assertEqual(vectors.shape, (3, 2))
        vectors[0, 0] = 100
        np.testing.assert_array_equal(store.get("third"), [5.0, 6.0])

    def test_get_batch_raises_for_missing_ids(self) -> None:
        store = VectorStore(2)
        store.add("first", np.array([1.0, 2.0]))

        with self.assertRaisesRegex(KeyError, "record ID does not exist: missing"):
            store.get_batch(["first", "missing"])

    def test_get_batch_empty_request_returns_empty_matrix(self) -> None:
        store = VectorStore(3)

        vectors = store.get_batch([])

        self.assertEqual(vectors.shape, (0, 3))
        self.assertEqual(vectors.dtype, np.dtype(np.float32))

    def test_get_all_active_returns_matching_copies_in_slot_order(self) -> None:
        store = VectorStore(2)
        store.add("first", np.array([1.0, 2.0]))
        store.add("second", np.array([3.0, 4.0]))

        ids, vectors = store.get_all_active()

        self.assertEqual(ids, ["first", "second"])
        np.testing.assert_array_equal(vectors, [[1.0, 2.0], [3.0, 4.0]])
        ids[0] = "changed"
        vectors[0, 0] = 100
        self.assertEqual(store.get_all_active()[0], ["first", "second"])
        np.testing.assert_array_equal(store.get("first"), [1.0, 2.0])

    def test_get_all_active_empty_store_returns_empty_matrix(self) -> None:
        store = VectorStore(3)

        ids, vectors = store.get_all_active()

        self.assertEqual(ids, [])
        self.assertEqual(vectors.shape, (0, 3))
        self.assertEqual(vectors.dtype, np.dtype(np.float32))

    def test_update_replaces_vector_without_changing_id_or_slot(self) -> None:
        store = VectorStore(2)
        store.add("first", np.array([1.0, 2.0]))
        store.add("second", np.array([3.0, 4.0]))
        mappings = store._id_to_slot.copy()
        slot_ids = store._slot_to_id.copy()

        store.update("first", np.array([7.0, 8.0]))

        np.testing.assert_array_equal(store.get("first"), [7.0, 8.0])
        self.assertEqual(store._id_to_slot, mappings)
        self.assertEqual(store._slot_to_id, slot_ids)
        self.assertEqual(store.count(), 2)
        store._assert_invariants()

    def test_update_missing_id_raises_key_error(self) -> None:
        store = VectorStore(2)

        with self.assertRaisesRegex(KeyError, "record ID does not exist: missing"):
            store.update("missing", np.array([1.0, 2.0]))

        self.assertEqual(store.count(), 0)
        store._assert_invariants()

    def test_invalid_update_leaves_original_vector_and_mappings_unchanged(self) -> None:
        store = VectorStore(2)
        store.add("first", np.array([1.0, 2.0]))
        original_mappings = store._id_to_slot.copy()
        original_slot_ids = store._slot_to_id.copy()
        invalid_vectors = (
            (np.array([3.0]), ValueError),
            (np.array([np.nan, 3.0]), ValueError),
            (np.array([np.inf, 3.0]), ValueError),
            (np.array([1e100, 3.0]), ValueError),
            (np.array([1.0 + 2.0j, 3.0]), TypeError),
            (np.array([True, False]), TypeError),
        )

        for vector, error in invalid_vectors:
            with self.subTest(vector=vector):
                with self.assertRaises(error):
                    store.update("first", vector)
                np.testing.assert_array_equal(store.get("first"), [1.0, 2.0])
                self.assertEqual(store._id_to_slot, original_mappings)
                self.assertEqual(store._slot_to_id, original_slot_ids)
                self.assertEqual(store.count(), 1)
                store._assert_invariants()

    def test_update_copies_replacement_from_caller_array(self) -> None:
        store = VectorStore(2)
        store.add("first", np.array([1.0, 2.0]))
        replacement = np.array([3.0, 4.0])

        store.update("first", replacement)
        replacement[0] = 100.0

        np.testing.assert_array_equal(store.get("first"), [3.0, 4.0])

    def test_delete_first_record_moves_last_record_and_preserves_its_vector(self) -> None:
        store = VectorStore(2)
        store.add("first", np.array([1.0, 2.0]))
        store.add("middle", np.array([3.0, 4.0]))
        store.add("last", np.array([5.0, 6.0]))

        store.delete("first")

        self.assertEqual(store.count(), 2)
        self.assertFalse(store.contains("first"))
        self.assertEqual(store._id_to_slot, {"last": 0, "middle": 1})
        self.assertEqual(store._slot_to_id, ["last", "middle"])
        np.testing.assert_array_equal(store.get("last"), [5.0, 6.0])
        self.assertEqual(store.get_all_active()[0], ["last", "middle"])
        store._assert_invariants()

    def test_delete_middle_record_moves_last_record_to_deleted_slot(self) -> None:
        store = VectorStore(2)
        for record_id, vector in (
            ("A", [1.0, 2.0]),
            ("B", [3.0, 4.0]),
            ("C", [5.0, 6.0]),
            ("D", [7.0, 8.0]),
        ):
            store.add(record_id, np.array(vector))

        store.delete("B")

        self.assertEqual(store.count(), 3)
        self.assertFalse(store.contains("B"))
        self.assertEqual(store._slot_to_id, ["A", "D", "C"])
        self.assertEqual(store._id_to_slot, {"A": 0, "D": 1, "C": 2})
        np.testing.assert_array_equal(store.get("D"), [7.0, 8.0])
        self.assertNotIn("B", store.get_all_active()[0])
        store._assert_invariants()

    def test_delete_final_record_does_not_move_other_records(self) -> None:
        store = VectorStore(2)
        store.add("first", np.array([1.0, 2.0]))
        store.add("last", np.array([3.0, 4.0]))

        store.delete("last")

        self.assertEqual(store.count(), 1)
        self.assertEqual(store._id_to_slot, {"first": 0})
        self.assertEqual(store._slot_to_id, ["first"])
        np.testing.assert_array_equal(store.get("first"), [1.0, 2.0])
        store._assert_invariants()

    def test_delete_only_record_leaves_empty_active_storage(self) -> None:
        store = VectorStore(2)
        store.add("only", np.array([1.0, 2.0]))

        store.delete("only")

        self.assertEqual(store.count(), 0)
        self.assertFalse(store.contains("only"))
        self.assertEqual(store._id_to_slot, {})
        self.assertEqual(store._slot_to_id, [])
        ids, vectors = store.get_all_active()
        self.assertEqual(ids, [])
        self.assertEqual(vectors.shape, (0, 2))
        store._assert_invariants()

    def test_delete_missing_record_raises_without_changing_store(self) -> None:
        store = VectorStore(2)
        store.add("existing", np.array([1.0, 2.0]))

        with self.assertRaisesRegex(KeyError, "record ID does not exist: missing"):
            store.delete("missing")

        self.assertEqual(store.count(), 1)
        self.assertEqual(store._id_to_slot, {"existing": 0})
        self.assertEqual(store._slot_to_id, ["existing"])
        np.testing.assert_array_equal(store.get("existing"), [1.0, 2.0])
        store._assert_invariants()


if __name__ == "__main__":
    unittest.main()
