import numpy as np
from knn_approximated import make_groups, get_centers

N = 100
input_matrix = np.load('../data/norm_decaf.npy')

res = []
cs = []

for i in range(N):
	print i
	data, centers = get_centers(input_matrix)
	groups = make_groups(data, centers, 10, 2*len(centers), [])
	c = 0
	for g1 in groups:
		for g2 in groups:
			if g1.id == g2.id : continue
			if np.sum(np.abs(g1.center - g2.center)) <= g1.r + g2.r:
				c += 1
	bof = (2.0/ (len(centers)*(len(centers)-1))) * c
	res.append(bof)
	cs.append(c)
	
print res
print c
x = open('bof.txt', 'w')
x.write(','.join([str(i) for i in res]))
x.close()