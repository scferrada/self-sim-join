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
	#return data[idx[0], :], data[idx[1], :]
	return data[idx[0]], data[idx[1]]

def qpartition(data, p, r, rho):
	dists = np.sum(np.abs(data[:,1:] - p[1:]), axis=1)
	L = data[dists < rho]
	G = data[dists >= rho]
	Gw = data[np.logical_and(dists>=rho, dists <= (rho + r))]
	Lw = data[np.logical_and(dists>=rho-r, dists <rho)]
	return L, G, Lw, Gw

def piv_join(data1, data2, results, r, k, eps):
	N = k
	if N > data1.shape[0]:
		N = data1.shape[0]
	P = []
	distances = 0
	for i in range(N):
		P.append(np.sum(np.abs(data2[:,1:] - data1[i][1:]), axis = 1))
		distances += data2.shape[0]
		for j, dist in enumerate(P[i]):
			if data1[i][0]==data2[j][0]: continue
			if data2[j][0] in [x.obj for x in results[data1[i][0]]]: continue
			if len(results[data1[i][0]]) < k:
				heappush(results[data1[i][0]], Res(data2[j][0], dist))
			elif dist < -results[data1[i][0]][0].dist:
				heappushpop(results[data1[i][0]], Res(data2[j][0], dist))
	for i in range(N, data1.shape[0]):
		dists = np.sum(np.abs(data1[:N, 1:] - data1[i][1:]), axis=1)
		distances += k
		for j in range(data2.shape[0]):
			if data1[i][0]==data2[j][0]: continue
			if data2[j][0] in [x.obj for x in results[data1[i][0]]]: continue
			f = False
			for l in range(N):
				if np.abs(P[l][j] - dists[l]) >  r:
					f = True
					break
			if not f: 
				dist  = np.sum(np.abs(data1[i][1:]-data2[j][1:]))
				distances += 1
				if len(results[data1[i][0]]) < k:
					heappush(results[data1[i][0]], Res(data2[j][0], dist))
				elif dist < -results[data1[i][0]][0].dist:
					heappushpop(results[data1[i][0]], Res(data2[j][0], dist))
	return results, distances, None
	
def knn_join(data1, data2, results, r, k, eps):
	d = 0
	maxdist = 0
	for elem_i in data1:
		distances = np.sum(np.abs(data2[:,1:]-elem_i[1:]), axis=1)
		d += len(data2)
		for i, dist in enumerate(distances):
			if data2[i][0] == elem_i[0]: continue
			if dist > maxdist: maxdist = dist
			if len(results[elem_i[0]]) < k:
				heappush(results[elem_i[0]], Res(data2[i][0], dist))
			elif dist < -results[elem_i[0]][0].dist:
				heappushpop(results[elem_i[0]], Res(data2[i][0],dist))
	return results, d, maxdist
	
'''
Data is a numpy matrix, where every row represents a vector
r is the range query parameter
c is the minumum amount of vectors needed to perform another recursion step
'''

def quickjoin_iter(data, r, c, k, join_func, results = None, eps = 0, distances=0):
	q = Queue.Queue()
	qw = Queue.Queue()
	q.put(data)
	if results is None:
		results = {x[0]:[] for x in data}
	while not q.empty():
		slice = q.get()
		if slice.shape[0] <= c:
			results, p_dists, maxdist1 = join_func(slice, slice, results, r, k, eps)
			distances += p_dists
			q.task_done()
			continue
		p1, p2 = choose_pivots(slice)
		rho = np.sum(np.abs(p1[1:]-p2[1:]))
		distances += 1
		L, G, Lw, Gw = qpartition(slice, p1, r, rho)
		distances += slice.shape[0]
		q.task_done()
		q.put(L)
		q.put(G)
		if Lw.shape[0] != 0 or Gw.shape[0] != 0:
			qw.put((Lw, Gw))
	maxdist2 = -1
	while not qw.empty():
		Lw, Gw = qw.get()
		if (Lw.shape[0] + Gw.shape[0]) <= c:
			results, p_dists, maxdist2 = join_func(Lw, Gw, results, r, k, eps)
			distances += p_dists
			qw.task_done()
			continue
		p1, p2 = choose_pivots(np.vstack((Lw, Gw)))
		rho = np.sum(np.abs(p1[1:]-p2[1:]))
		distances += 1
		L1, G1, Lw1, Gw1 = qpartition(Lw, p1, r, rho)
		L2, G2, Lw2, Gw2 = qpartition(Gw, p1, r, rho)
		distances += Lw.shape[0] + Gw.shape[0]
		qw.task_done()
		qw.put((L1,L2))
		qw.put((G1,G2))
		qw.put((Lw1,Gw2))
		qw.put((Gw1,Lw2))
	q.join()
	qw.join()
	return results, distances, max(maxdist1, maxdist2)
	
def quickjoin(data, k, c):
	idx = np.arange(len(data)).reshape(len(data),1)
	data = np.hstack((idx, data))
	print("first run of QJ")
	results, distances, maxdist = quickjoin_iter(data, 0, c, k, knn_join)
	#return results, distances
	print("second run of QJ")
	results2, distances2, _ = quickjoin_iter(data, maxdist, c, k, piv_join, results)
	return results2, distances+distances2
	