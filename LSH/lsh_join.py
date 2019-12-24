import numpy as np

'''
Computes a LSH over the provided data:
data: a numpy matrix containing vectors as rows
h_k: the number of sensitive hash functions to use
g_l: the number of composed hash functions to use
r: the number of buckets to use

returns, the LS-hash table for the data and the hash functions
'''
def hash_data(data, h_k, g_l, r):
	N = data.shape[1]-1
	G = []
	for j in xrange(g_l):
		H = []
		for i in xrange(h_k):
			a = np.abs(np.random.standard_cauchy(N))
			b = np.random.uniform(low=0, high=r)
			H.append ([a,b])#(lambda x: np.floor((np.inner(x[1:], a)+b)/r))
		G.append(H)
	tables = [{} for _ in G]
	for v in data:
		for j, g_j in enumerate(G):
			coord = []
			for h_i in g_j:
				c = np.floor((np.inner(v[1:], h_i[0])+h_i[1])/r)
				coord.append(int(c))#(h_i(v))
			t_coord = tuple(coord)
			if t_coord not in tables[j]:
				tables[j][t_coord] = [int(v[0])]
			else:
				tables[j][t_coord].append(int(v[0]))
	return tables, G
	
def get_knn(tables, G, data, q, k, r, distance):
	candidates = []
	for i, table in enumerate(tables):
		q_hash = []
		for h_i in G[i]:
			c = np.floor((np.inner(q[1:], h_i[0])+h_i[1])/r)
			q_hash.append(c)
		if tuple(q_hash) in table:
			candidates.extend(table[tuple(q_hash)])
		if len(candidates) >= 3*len(tables):
			break
	ddd = np.unique([int(x) for x in candidates if x!= q[0]])
	#print ddd
	#print len(candidates)
	#print ddd.shape 
	#print q[0]
	candidates = data[[int(i) for i in ddd]]
	if len(candidates) <= k:
		return candidates[:,0]
	distances = distance(candidates, q)
	idx = np.argpartition(distances, k)[:k]
	return candidates[idx, 0]
	
def manhattan(mat, v):
	return np.sum(np.abs(mat[:,1:]-v[1:]), axis=1)
	
def self_sim_join(data, h_k, g_l, r, k, distance=manhattan):
	idx = np.arange(len(data)).reshape(len(data), 1)
	data = np.hstack((idx, data))
	tables, G = hash_data(data, h_k, g_l, r)
	results = {}
	for row in data:
		knn = get_knn(tables, G, data, row, k, r, distance)
		results[row[0]] = knn.tolist()
	return results