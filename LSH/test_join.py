import numpy as np
from lsh_join import *

data = np.load('data/norm_decaf.npy')
idx = np.arange(len(data)).reshape(len(data), 1)
data = np.hstack((idx, data))

results = self_sim_join(data, 10, 30, 200, 16)

with open('out/lsh.res', 'w') as out:
	for x in results:
		out.write("%d, %s\n"%(x, ','.join([str(i) for i in results[x]])))
		