import numpy as np
from Distance import euclidean_distance, k_nearest_neighbors_batch
from models import CollectionConfig, VectorRecord
from collection import Collection
from database import VectorDatabase

if __name__ == "__main__":

   db = VectorDatabase()

   test_collection=db.create_collection(name="test_collection", dimension=3, metric="euclidean", index_type="flat")

   test_collection.insert(VectorRecord(id="vec1", vector=np.array([1.0, 2.0, 3.0]), metadata={"info": "first vector"}))
   test_collection.insert(VectorRecord(id="vec2", vector=np.array([4.0, 5.0, 6.0]), metadata={"info": "second vector"}))    
   test_collection_3=db.create_collection(name="test_collection_3", dimension=3, metric="euclidean", index_type="flat")

   test_collection_2=db.get_collection(name="test_collection")

   print("Collections in the database:", db.list_collections())
   print("Collection test_collection contains vec1:", test_collection.contains("vec1"))
   print("Collection test_collection_2 contains vec1:", test_collection_2.contains("vec1"))
   db.delete_collection(name="test_collection_3")
   print(db.count_collection())



   