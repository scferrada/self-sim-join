import numpy as np
import quickjoin as qj

N=100
x = np.arange(N).reshape(N,1)
data = np.hstack((x, np.hstack((x, x))))

p1 = data[15]
p2 = data[69]

rho = np.sum(np.abs(p1[1:]-p2[1:]))
print p1, p2, rho

L, G, Lw, Gw = qj.qpartition(data, p1, 0, rho)

print L.shape, G.shape, Lw.shape, Gw.shape
print Gw