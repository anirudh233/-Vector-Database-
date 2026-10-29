import numpy as np
from VECTOR_DATABASE.vectordb.core.Distance import euclidean_distance, k_nearest_neighbors
from VECTOR_DATABASE.vectordb.core.models import CollectionConfig, VectorRecord
from VECTOR_DATABASE.vectordb.core.collection import Collection
from VECTOR_DATABASE.vectordb.core.database import VectorDatabase

if __name__ == "__main__":
    db = VectorDatabase()

    collection = db.create_collection(
    name="test",
    dimension=2,
    metric="euclidean"
    )

    collection.insert(
    VectorRecord(
        id="vec_1",
        vector=np.array([1, 2]),
        metadata={}
                )
                      )

    collection.insert(
    VectorRecord(
        id="vec_2",
        vector=np.array([3, 4]),
        metadata={}
                 )
                 ) 

    vectors_name = collection.list_ids()
    vector = []
    for record_id in vectors_name:
        record = collection.get(record_id)
        vector.append(record.vector)

    vectors = np.array(vector)
    query_vector = np.array([[4, 3]])
    arr=k_nearest_neighbors(query_vector, vectors, k=2, kind="euclidean") 
    print("Nearest neighbor indices: ", arr)

        