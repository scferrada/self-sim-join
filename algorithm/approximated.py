import os, math
import numpy as np 

def get_centers(data):
	h, w = data.shape
	idx = np.random.choice(h, size=math.ceil(math.sqrt(h)), replace=False)
	return data[idx, :]

def make_groups(data, centers, max_size):
	results = {x:[] for x, _ in enumerate(centers)}
	i = 0
	for v in data:
		distances = np.abs(np.sum(centers - v, axis=1))
		indices = np.argsort(distances)
		for index in indices:
			if len(results[index]) < max_size:
				results[index].append((i,v))
				i += 1
				break
	return results	

def sim_join(input_matrix, output_folder, factor=1, iteration=0):	
	data = np.load(input_matrix)
	centers = get_centers(data)
	groups = make_groups(data, centers, factor*centers.shape[0])
	with open(os.path.join(output_folder, iteration), 'w') as outfile:	
		for group in groups:
			for i, x in groups[group]:
				min_dist = float("inf")
				nn = i
				for j, y in groups[group]:
					if np.array_equal(x, y): continue
					dist = np.abs(np.sum(x-y))
					if dist < min_dist:
						min_dist = dist
						nn = j
				outfile.write("%d,%d\n" % (i, nn))