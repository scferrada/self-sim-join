from lsh_join import hash_data
import numpy as np

N = 100000
l = []

data = np.load('data/dataset.npy')[:100000]
idx = np.arange(len(data)).reshape(len(data), 1)
data = np.hstack((idx, data))

table, G = hash_data(data, 10, 30, 18)

less = 0
more = 0
print len(table)
for x in table:
	for y in x:
		c = len(x[y])
		if c>1:
			more += 1
		else:
			less += 1
print more
print less