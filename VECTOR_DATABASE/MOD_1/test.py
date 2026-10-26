import numpy as np
from Distance import euclidean_distance, k_nearest_neighbors_batch






if __name__ == "__main__":
    a=np.array([[1, 2], [3, 4],[5,7],[0,0],[1,1],[2,2],[3,9]])
    b=np.array([[12,1], [32,4],[5,3],[2,6],[1,-2],[-2,2],[-3,0]])
    distance_eq = euclidean_distance(a,b)
    print(distance_eq)
    print('k neighbours are ',k_nearest_neighbors_batch(a,b,3, batch_size=2, kind="euclidean"))
    pass
