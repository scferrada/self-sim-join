from lsh_join import hash_data
import numpy as np

N = 100000
l = []

data = np.load('data/dataset.npy')
data = data[0:len(data)/10]
idx = np.arange(len(data)).reshape(len(data), 1)
data = np.hstack((idx, data))
k = 10
l = 30
r = 80

table, G = hash_data(data, k, l, r)

less = 0
more = 0
for x in table:
	for y in x:
		c = len(x[y])
		if c>1:
			more += 1
		else:
			less += 1
print more
print less
print k, l, r