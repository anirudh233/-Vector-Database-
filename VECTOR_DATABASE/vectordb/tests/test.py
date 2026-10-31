import numpy as np
from VECTOR_DATABASE.vectordb.core.Distance import euclidean_distance, k_nearest_neighbors
from VECTOR_DATABASE.vectordb.core.models import CollectionConfig, VectorRecord
from VECTOR_DATABASE.vectordb.core.collection import Collection
from VECTOR_DATABASE.vectordb.core.database import VectorDatabase

if __name__ == "__main__":


    arr =np.array([[1,3,4,5],[6,7,8,9]])

    arr_slice = arr[1,:]
    arr_slice[0] = 4 
    print('array parameters ',arr.shape,arr_slice.shape,arr.dtype,arr_slice.dtype,arr.nbytes,arr_slice.nbytes,arr.flags.c_contiguous,arr_slice.flags.c_contiguous,arr.strides,arr_slice.strides)
    print(arr_slice,arr)
