import numpy as np
import matplotlib.pyplot as plt

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
	idx = np.random.choice(h, size=np.ceil(np.sqrt(h)), replace=False)
	centers = input_matrix[idx, :]
	mask = np.ones(len(input_matrix), dtype=bool)
	mask[idx] = False
	data = input_matrix[mask, :]
	return data, centers

def make_groups(data, centers, max_size):
	groups = [Group(x, [], 0, id) for id, x in enumerate(centers)]
	slices = 1
	step = int(np.floor(len(data)/slices))
	for i in range(slices):
		D = np.abs((data[i*step:(i+1)*step,1:,None]-centers[:,1:,None].T)).sum(1)
		cg = np.argsort(D, axis=1)
		done = []
		for el, row in enumerate(data[i*step:(i+1)*step]):
			j = 0
			while True:
				closest_group = cg[el,j]
				if len(groups[closest_group]) < max_size:
					groups[closest_group].stack(row, D[el, closest_group])
					break
				j += 1
	return [g.consolidate() for g in groups]	

x = np.random.rand(100, 1)
y = np.random.rand(100, 1)

group_size = 1

input_matrix = np.hstack((x,y))
idx = np.arange(len(input_matrix)).reshape(len(input_matrix), 1)
matrix = np.hstack((idx, input_matrix))
data, centers = get_centers(matrix)
print data.shape
groups = make_groups(data, centers, group_size * np.ceil(np.sqrt(len(input_matrix))))

for g in groups:
	plt.scatter(g.elems[:,1], g.elems[:,2])
plt.show()