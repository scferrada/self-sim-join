import numpy as np
import math
from heapq import heappush, heappushpop, heapify

class Res:
	def __init__(self, obj, dist):
		self.dist = -dist
		self.obj = obj

	def __lt__(self, other):
		return self.dist < other.dist

	def __eq__(self, other):
		return self.dist == other.dist and self.obj == other.obj

	def __str__(self):
		return str(self.obj) + "; " + str(self.dist)


class Group:
	def __init__(self, center=None, elems=[], r=0, id=-1):
		self.center = center
		self.elems = elems
		self.id = id
		self.r = r
		self.consolidated = False
		self.c = 0

	def stack(self, element, dist):
		self.elems.append(element)
		if dist > self.r:
			self.r = dist

	def __len__(self):
		if self.center is None: return 0
		if self.elems is None: return 1
		if not self.consolidated: return len(self.elems)
		if len(self.elems.shape) == 1: return 2
		return len(self.elems) + 1

	def show(self):
		print(self.all())

	def get_elems(self):
		return self.elems
	
	def consolidate(self):
		if len(self.elems)==0:
			self.elems = None
			self.consolidated = True
			return self
		temp = np.empty((len(self.elems), self.elems[0].shape[0]))
		for i, x in enumerate(self.elems):
			temp[i,:] = x
		self.elems = temp		
		self.consolidated = True
		return self

	def all(self):
		if len(self) == 1: return self.center
		return np.concatenate((self.elems, self.center.reshape((1,len(self.center)))))

	def all_but(self, index):
		if len(self) == 1: return self.center
		if len(self) == 2 and index == 0: return self.center
		return np.concatenate((self.elems[:index], self.elems[index + 1:], self.center.reshape((1,len(self.center)))))


def get_centers(input_matrix):
	h, w = input_matrix.shape
	idx = np.random.choice(h, size=math.ceil(math.sqrt(h)), replace=False)
	centers = input_matrix[idx, :]
	mask = np.ones(len(input_matrix), dtype=bool)
	mask[idx] = False
	data = input_matrix[mask, :]
	return data, centers


def get_better_centers(input_matrix):
	h, w = input_matrix.shape
	idx = np.random.choice(h, size=2*math.ceil(math.sqrt(h)), replace=False)
	candidates = input_matrix[idx, :]
	distances = distance_matrix(candidates, candidates, 1)
	MAXDIST = distances.max()
	center_idx = [0]
	for i, candidate in enumerate(candidates[1:]):
		add = True
		for center in center_idx:
			if distances[center][i] < MAXDIST*0.07:
				add = False
				break
		if add:
			center_idx.append(i)
	idx = np.array([int(candidates[x][0]) for x in center_idx])
	centers = input_matrix[idx, :]
	mask = np.ones(len(input_matrix), dtype=bool)
	mask[idx] = False
	data = input_matrix[mask, :]
	return data, centers
	
def get_vp_centers(data, c):
	class VP:
		def __init__(self, point, spread):
			self.point = point
			self.spread = spread
		def __lt__(self, other):
			return self.spread < other.spread
			
	N = int(max(np.ceil(np.sqrt(len(data))*c), 100))
	P = data[np.random.choice(data.shape[0], N, replace=False)]
	vpoints = [VP(x[0], 0) for x in P[0:np.ceil(np.sqrt(len(data)))]]
	for pi in P:
		D = data[np.random.choice(data.shape[0], N, replace=False)]
		distances = np.sum(np.abs(D[:,1:]-pi[1:]), axis=1)
		mu = np.median(distances)
		spread = np.std(distances - mu)
		if spread > vpoints[0].spread:
			heappushpop(vpoints, VP(pi[0], spread))
	idx = [x.point for x in vpoints]
	return np.delete(data, idx, axis=0), data[idx]


def make_groups(data, centers, k, max_size, results):
	groups = [Group(x, [], 0, id) for id, x in enumerate(centers)]
	slices = 500000
	step = int(math.floor(len(data)/slices))
	for i in range(slices):
		D = np.abs((data[i*step:(i+1)*step,1:,None]-centers[:,1:,None].T)).sum(1)
		k1 = min(step-1, k)
		idx = np.argpartition(D, k1, axis=1)[:, :k1]
		center_nn = np.argpartition(D.T, k1, axis=1)[:, :k1]
		for j, el in enumerate(data[i*step:(i+1)*step]):
			knn_dist = [Res(nn, d) for nn, d in zip(idx[j], D[j, idx[j]])]
			if len(results[el[0]]) == 0:
				heapify(knn_dist)
				results[el[0]] = knn_dist
				continue
			for r in knn_dist:
				if len(results[el[0]]) < k:
					heappush(results[el[0]], r)
				elif results[el[0]][0] > -r.dist:	
					heappushpop(results[el[0]], r)
		for j, cent in enumerate(centers):
			knn_dist = [Res(nn, d) for nn, d in zip(center_nn[j], D[center_nn[j], j])]
			if i==0:
				heapify(knn_dist)
				results[cent[0]] = knn_dist
				continue
			for r in knn_dist:
				if len(results[cent[0]]) < k:
					heappush(results[cent[0]], r)
				elif results[cent[0]][0]>-r.dist:
					heappushpop(results[cent[0]], r)
		cg = np.argsort(D, axis=1)
		for el, row in enumerate(data[i*step:(i+1)*step]):
			j = 0
			while True:
				closest_group = cg[el,j]
				if len(groups[closest_group]) < max_size:
					groups[closest_group].stack(row, D[el, closest_group])
					break
				j += 1
	return [g.consolidate() for g in groups]			

def sim_join(input_matrix, k, group_size=1):
	idx = np.arange(len(input_matrix)).reshape(len(input_matrix), 1)
	matrix = np.hstack((idx, input_matrix))
	results = {x[0]: [] for x in matrix}
	data, centers = get_centers(matrix)
	#data, centers = get_vp_centers(matrix, group_size)#, results, k)
	print("making %d, %d *sqrt(n)-sized groups..." % (len(centers), group_size))
	groups = make_groups(data, centers, k, group_size * math.ceil(math.sqrt(len(input_matrix))), results)
	R = np.array([group.r for group in groups])
	print("nested loop... %d groups" % len(groups))
	for i, group in enumerate(groups):
		if len(group) <= 2:
			continue
		else:
			g = group.elems
		for current, elem_i in enumerate(g):
			dist_to_groups = (np.sum(np.abs(centers[:, 1:] - elem_i[1:]), axis=1) - R)
			j = 0
			while True:
				closest_group = np.argpartition(dist_to_groups, j)[j]
				if groups[closest_group].id != group.id:
					break
				j += 1
			target = np.vstack((group.all_but(current), groups[closest_group].all()))
			while len(target) <= k:
				j += 1
				closest_group = np.argpartition(dist_to_groups, j)[j]
				if groups[closest_group].id == group.id:
					continue
				target = np.vstack((target, groups[closest_group].all()))
			distances = np.sum(np.abs(target[:, 1:] - elem_i[1:]), axis=1)
			idx = np.argpartition(distances, k)[:k]
			knn = target[idx]
			for d, e in enumerate(knn):
				if len(results[elem_i[0]])< k:
					heappush(results[elem_i[0]], Res(e[0], distances[idx][d]))
				elif distances[idx][d] < -results[elem_i[0]][0].dist:
					heappushpop(results[elem_i[0]], Res(e[0], distances[idx][d]))
	return results
