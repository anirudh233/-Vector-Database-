from scipy.spatial import distance
import numpy as np
coords = [(3,4),(4,3)]
coords2 = [(1,2),(4,3),(10,9)]
print(distance.cdist(coords, coords2, 'sqeuclidean'))