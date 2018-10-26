import knn_approximated as knn
import numpy as np

N = 25
l = []
for i in range(N):
    l.append([i, i])

data = np.array(l)

res = knn.sim_join(data, 2, 2)