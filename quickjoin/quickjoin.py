import os, math
import numpy as np

'''
This is the approximated version of Quickjoin, as seen in "Quicker Similarity Joins in Metric Spaces"
by Fredriksson and Braithwaite
'''

def choose_pivots(data):
	idx = np.random.choice(data.shape[0], size=2, replace=False)
	return data[idx[0], :], data[idx[1], :]

def qpartition(data, p, r, rho):
	dists = np.sum(np.abs(data - p), axis=1))
	L = data[dists < rho]
	G = data[dists >= rho]
	Gw = data[dists <= (rho + r)]
	Lw = L[dists >= rho - r]
	return L, G, Lw, Gw
	
def piv_join(data1, data2, r, k, eps):
	results = []
	P = []
	distances = 0
	for i in range(k):
		P.append(np.sum(np.abs(data2 - data1[i]), axis = 1))
		distances += data2.shape[0]
		results.extend([(data1[i], x) for j,x in enumerate(data2) if P[i][j] < r])
	for i in range(k, data1.shape[0]):
		dists = np.sum(np.abs(data1[:k] - data1[i]), axis=1)
		distances += k
		for j in range(data2.shape[0]):
			f = False
			for l in range(k):
				distances += 1
				if math.abs(P[l][j] - dists[l]) > (1 - eps) * r:
					f = True
					break
			if not f and np.sum(np.abs(data1[i]-data2[j])) <= r:
				distances += 1
				results.append((data1[i], data2[j]))
	return results, distances
'''
Data is a numpy matrix, where every row represents a vector
r is the range query parameter
c is the minumum amount of vectors needed to perform another recursion step
'''
def quickjoin(data, r, c, k, eps = 0, distances=0):
	if data.shape[0] <= c:
		results, p_dists = piv_join(data, data, r, k, eps)
		distances += p_dists
		return results, distances
	p1, p2 = choose_pivots(data)
	rho = np.sum(np.abs(p1-p2))
	distances += 1
	L, G, Lw, Gw = qpartition(data, p1, r, rho)
	distances += data.shape[0]
	results = []
	r, d = quickjoin(L, r, c, k, eps, distances)
	results.extend(r)
	r, d = quickjoin(G, r, c, k, eps, d)
	results.extend(r)
	r, d = quickjoin_win(Lw, Gw, r, c, k, eps)
	results.extend(r)
	return results, d
	
def quickjoin_win(Lw, Gw, r, c, k, eps, distances):
	if (Lw.shape[0] + Gw.shape[0]) <= c:
		res, p_dists piv_join(Lw, Gw, r, k, eps)
		distances += p_dists
		return res, distances
	p1, p2 = choose_pivots(np.vstack((Lw, Gw)))
	rho = np.sum(np.abs(p1-p2))
	distances += 1
	L1, G1, Lw1, Gw1 = qpartition(Lw, p1, r, rho)
	L2, G2, Lw2, Gw2 = qpartition(Gw, p1, r, rho)
	results = []
	r, d = quickjoin_win(L1, L2, r, c, k, eps, distances)
	results.extend(r)
	r, d = quickjoin_win(G1, G2, r, c, k, eps, d)
	results.extend(r)
	r, d = quickjoin_win(Lw1, Gw2, r, c, k, eps, d)
	results.extend(r)
	r, d = quickjoin_win(Gw1, Lw2, r, c, k, eps, d)
	results.extend(r)
	return results, d
	