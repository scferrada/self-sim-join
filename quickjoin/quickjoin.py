import os, math, Queue
import numpy as np
from heapq import heappush, heappop, heappushpop

'''
This is the approximated version of Quickjoin, as seen in "Quicker Similarity Joins in Metric Spaces"
by Fredriksson and Braithwaite
'''

class Res:
	def __init__(self, obj, dist):
		self.dist = -dist
		self.obj = obj

	def __lt__(self, other):
		return self.dist < other.dist
		
	def __leq__(self, other):
		return self.dist <= other.dist

	def __eq__(self, other):
		return self.dist == other.dist and self.obj == other.obj

	def __str__(self):
		return str(self.obj) + "; " + str(self.dist)


def choose_pivots(data):
	idx = np.random.choice(data.shape[0], size=2, replace=False)
	return data[idx[0]], data[idx[1]]
	
def choose_better_pivots(data, c):
	x = np.ceil(len(data)*0.05)
	if x < c: 
		x = c
	N = int(min(x, 100))
	P = data[np.random.choice(data.shape[0], N, replace=False)]
	best_spread = 0
	min_spread = np.float('inf')
	p1 = P[0]
	p2 = P[0]
	for pi in P[1:]:
		D = data[np.random.choice(data.shape[0], N, replace=False)]
		distances = np.sum(np.abs((D[:,1:] - pi[1:])), axis=1)
		spread = np.var(distances)
		if spread > best_spread:
			best_spread = spread
			p1 = pi
		elif spread < min_spread:
			min_spread = spread
			p2 = pi
	return p1, p2


#TODO: test
def qpartition(data, p, r, rho):
	dists = np.sum(np.abs(data[:, 1:] - p[1:]), axis=1)
	L = data[dists < rho]
	G = data[dists >= rho]
	Gw = data[np.logical_and(dists >= rho, dists <= (rho + r))]
	Lw = data[np.logical_and(dists >= rho - r, dists < rho)]
	return L, G, Lw, Gw


def piv_join(data1, data2, results, r, k, eps):
	print "nope"
	exit()
	N = k
	if N > data1.shape[0]:
		N = data1.shape[0]
	P = []
	distances = 0
	for i in range(N):
		P.append(np.sum(np.abs(data2[:, 1:] - data1[i][1:]), axis=1))
		distances += data2.shape[0]
		for j, dist in enumerate(P[i]):
			if data1[i][0] == data2[j][0]: continue
			if data2[j][0] in [x.obj for x in results[data1[i][0]]]: continue
			if len(results[data1[i][0]]) < k:
				heappush(results[data1[i][0]], Res(data2[j][0], dist))
			elif dist < results[data1[i][0]][0].dist:
				heappushpop(results[data1[i][0]], Res(data2[j][0], dist))
	for i in range(N, data1.shape[0]):
		dists = np.sum(np.abs(data1[:N, 1:] - data1[i][1:]), axis=1)
		distances += k
		for j in range(data2.shape[0]):
			if data1[i][0] == data2[j][0]: continue
			if data2[j][0] in [x.obj for x in results[data1[i][0]]]: continue
			f = False
			for l in range(N):
				if np.abs(P[l][j] - dists[l]) > r:
					f = True
					break
			if not f:
				dist = np.sum(np.abs(data1[i][1:] - data2[j][1:]))
				distances += 1
				if len(results[data1[i][0]]) < k:
					heappush(results[data1[i][0]], Res(data2[j][0], dist))
				elif dist < results[data1[i][0]][0].dist:
					heappushpop(results[data1[i][0]], Res(data2[j][0], dist))
	return results, distances, None


def knn_join(data1, data2, results, r, k, eps):
	if len(data2) == 0:
		return results, 0, r
	distances = np.abs(data1[:,1:,None]-data2[:,1:,None].T).sum(1)
	d = len(data2) * len(data1)
	if len(data2) <= k+1:
		idx = np.argsort(distances, axis=1)
	else:	
		idx = np.argpartition(distances, k+1, axis=1)[:, :k+1]
	for i in range(len(data1)):
		for j in range(min(k+1,len(data2))):
			if data1[i][0] == data2[idx[i][j]][0]: continue
			dist = distances[i,idx[i][j]]
			if len(results[data1[i][0]]) < k:
				heappush(results[data1[i][0]], Res(data2[idx[i][j]][0], dist))
			elif dist < -results[data1[i][0]][0].dist:
				heappushpop(results[data1[i][0]], Res(data2[idx[i][j]][0], dist))
	if r!=0 and all([len(results[x])==k for x in results]):
		d = - min([x[0].dist for x in results.values()])
		if d<r:
			r = d
	return results, d, r

				
'''
Data is a numpy matrix, where every row represents a vector
r is the range query parameter
c is the minimum amount of vectors needed to perform another recursion step
'''
def quickjoin_iter(data, r, c, k, join_func, results=None, eps=0, distances=0):
	q = Queue.LifoQueue()
	qw = Queue.LifoQueue()
	q.put(data)
	if results is None:
		results = {x[0]: [] for x in data}
	while not q.empty():
		slice = q.get()
		if slice.shape[0] <= c:
			results, p_dists, _ = join_func(slice, slice, results, r, k, eps)
			distances += p_dists
			q.task_done()
			continue
		p1, p2 = choose_pivots(slice)
		rho = np.sum(np.abs(p1[1:] - p2[1:]))
		distances += 1
		L, G, Lw, Gw = qpartition(slice, p1, r, rho)
		distances += slice.shape[0]
		q.task_done()
		q.put(L)
		q.put(G)
		if r == 0: continue
		qw.put((Lw, Gw))
	while not qw.empty():
		Lw, Gw = qw.get()
		if (Lw.shape[0] + Gw.shape[0]) <= 2*c:
			results, p_dists, r = join_func(Lw, Gw, results, r, k, eps)
			distances += p_dists
			qw.task_done()
			continue
		p1, p2 = choose_pivots(np.vstack((Lw, Gw)))
		rho = np.sum(np.abs(p1[1:] - p2[1:]))
		distances += 1
		L1, G1, Lw1, Gw1 = qpartition(Lw, p1, r, rho)
		L2, G2, Lw2, Gw2 = qpartition(Gw, p1, r, rho)
		distances += Lw.shape[0] + Gw.shape[0]
		qw.task_done()
		if L1.shape[0] != 0 and L2.shape[0] != 0:
			qw.put((L1, L2))
		if G1.shape[0] != 0 and G2.shape[0] != 0:
			qw.put((G1, G2))
		if Lw1.shape[0] != 0 and Gw2.shape[0] != 0:	
			qw.put((Lw1, Gw2))
		if Gw1.shape[0] != 0 and Lw2.shape[0] != 0:	
			qw.put((Gw1, Lw2))
	q.join()
	qw.join()
	return results, distances


def quickjoin(data, k, c):
	idx = np.arange(len(data)).reshape(len(data), 1)
	data = np.hstack((idx, data))
	print("first run of QJ")
	results, distances = quickjoin_iter(data, 0, c, k, knn_join)
	maxdist = - min([results[x][0].dist for x in results if len(results[x]) > 0])
	print("second run of QJ")
	results2, distances2 = quickjoin_iter(data, maxdist, c*10, k, knn_join, results)
	return results2, distances + distances2
