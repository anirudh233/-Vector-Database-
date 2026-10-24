import numpy as np
from scipy.spatial import distance

from Distance import cosine_similarity
from Distance import cosine_distance
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from sklearn.metrics.pairwise import cosine_distances

X = [[0, 0, 0], [1, 1, 1]]
Y = [[1, 0, 0], [1, 1, 0],[4,4,4], [1,1,1],[0,0,0], [0,1,0],[0,0,1],[1,0,1],[1,1,0],[0,1,1]]



def euclidean_distance(vectors1:np.ndarray, vectors2:np.ndarray) -> np.ndarray:
        vectors1 = np.asarray(vectors1, dtype=np.float32)
        vectors2 = np.asarray(vectors2, dtype=np.float32)

        if vectors1.ndim != 2 or vectors2.ndim != 2:
            raise ValueError("Expected 2d Matrix of Vectors")

        if not np.isfinite(vectors1).all() or not np.isfinite(vectors2).all():
            raise ValueError("Vector Contains Nan or Infinity")

        diff = vectors1[:, np.newaxis, :] - vectors2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

def euclidean_distance_2(vectors1:np.ndarray, vectors2:np.ndarray) -> np.ndarray:

       vectors1 =np.asarray(vectors1, dtype=np.float64)
       vectors2 =np.asarray(vectors2, dtype=np.float64)

       if vectors1.ndim !=2 or vectors2.ndim !=2:
             raise ValueError("Expected 2d Matrix of Vectors")

       if not np.isfinite(vectors1).all() or not np.isfinite(vectors2).all():
                raise ValueError("Vector Contains Nan or Infinity")
       
       if vectors1.shape[1] != vectors2.shape[1]:
                raise ValueError("Vectors must have the same dimension")
       
       m1,n1=vectors1.shape
       m2,_=vectors2.shape
       distances=np.zeros((m1,m2),dtype=np.float64)
       for i in range(m1):
            for j in range(m2):
                total =0.0
                for count in range(n1):
                    total += np.square(vectors1[i][count] - vectors2[j][count])
                distances[i][j] = np.sqrt(total)
       return distances        

def euclidean_distance_3(vectors1:np.ndarray, vectors2:np.ndarray,flag:bool=False) -> np.ndarray:

       vectors1 =np.asarray(vectors1, dtype=np.float64)
       vectors2 =np.asarray(vectors2, dtype=np.float64)

       if vectors1.ndim !=2 or vectors2.ndim !=2:
             raise ValueError("Expected 2d Matrix of Vectors")

       if not np.isfinite(vectors1).all() or not np.isfinite(vectors2).all():
                raise ValueError("Vector Contains Nan or Infinity")
       
       if vectors1.shape[1] != vectors2.shape[1]:
                raise ValueError("Vectors must have the same dimension")

       if flag:
            diff = vectors1[:, np.newaxis, :] - vectors2[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=-1))
       else:
             
        m1,n1=vectors1.shape
        m2,_=vectors2.shape
        distances=np.zeros((m1,m2),dtype=np.float64)
        for i in range(m1):
            for j in range(m2):
                total =0.0
                for count in range(n1):
                    total += np.square(vectors1[i][count] - vectors2[j][count])
                distances[i][j] = np.sqrt(total)
        return distances  

def sqeuclidean_distance(vectors1:np.ndarray, vectors2:np.ndarray,flag:bool=False) -> np.ndarray:

       vectors1 =np.asarray(vectors1, dtype=np.float64)
       vectors2 =np.asarray(vectors2, dtype=np.float64)

       if vectors1.ndim !=2 or vectors2.ndim !=2:
             raise ValueError("Expected 2d Matrix of Vectors")

       if not np.isfinite(vectors1).all() or not np.isfinite(vectors2).all():
                raise ValueError("Vector Contains Nan or Infinity")
       
       if vectors1.shape[1] != vectors2.shape[1]:
                raise ValueError("Vectors must have the same dimension")

       if flag:
            diff = vectors1[:, np.newaxis, :] - vectors2[np.newaxis, :, :]
            return (np.sum(diff ** 2, axis=-1))
       else:
             
        m1,n1=vectors1.shape
        m2,_=vectors2.shape
        distances=np.zeros((m1,m2),dtype=np.float64)
        for i in range(m1):
            for j in range(m2):
                total =0.0
                for count in range(n1):
                    total += np.square(vectors1[i][count] - vectors2[j][count])
                distances[i][j] = total
        return distances  

from Distance import k_nearest_neighbors

if __name__ == "__main__":
    arr=k_nearest_neighbors(X, Y, k=2)
    print(arr[0].shape)
    print(arr[1])
    print(arr)
    pass
