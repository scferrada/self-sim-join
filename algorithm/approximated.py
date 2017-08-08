import argparse, os, math
import numpy as np 

parser = argparse.ArgumentParser(description='Computes the approximated self similarity join (1NN) of a given ser of points')

parser.add_argument('input_matrix', type=str, help='the numpy vector storing file')
parser.add_argument('output_folder', type=str, help='the directory where the results must be stored')

args = parser.parse_args()

def get_centers(data):
	h, w = data.shape
	idx = np.random.randint(h, size=math.ceil(math.sqrt(h)), replace=False)
	return data[idx, :]

def make_groups(data, centers, max_size):
	results = {x:[] for x, _ in enumerate(centers)}
	i = 0
	for v in data:
		distances = np.abs(np.sum(centers - v, axis=1))
		indices = np.argsort(distances)
		for index in indices:
			if len(result[index]) < max_size:
				result[index].append((i,v))
				i += 1
				break
	return results	
	
data = np.load(args.input_matrix)
centers = get_centers(data)
groups = make_groups(data, centers, centers.shape[0])
with open(os.path.join(args.output_file), 'w') as outfile:	
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