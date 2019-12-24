import argparse, os, math
import numpy as np 
from heapq import heappush, heappushpop, heapify

parser = argparse.ArgumentParser(description='Computes self similarity join (kNN) of a given ser of points')

parser.add_argument('input_numpy', type=str, help='the numpy vectors')
parser.add_argument('output_folder', type=str, help='the directory where the results must be stored')
parser.add_argument('--batch', dest='batch', type=int, default='100', help='the percentage of data to be used')

args = parser.parse_args()

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

k = 16
dataset = np.load(args.input_numpy)[:2000000,:]

print "starting bruteforce for %d" % dataset.shape[0]
with open(os.path.join(args.output_folder, "ghdknn.csv"), "w") as outfile:	
	idx = np.arange(len(dataset)).reshape(len(dataset), 1)
	matrix = np.hstack((idx, dataset))	
	slices = 500000
	step = int(math.floor(len(matrix)/slices))
	for i in range(slices):
		D = np.abs((matrix[i*step:(i+1)*step,1:,None]-matrix[:,1:,None].T)).sum(1)
		idx_knn = np.argpartition(D,k, axis=1)[:, :k+1]
		for j, el in enumerate(matrix[i*step:(i+1)*step]):
			knn_dist = [Res(nn, d) for nn, d in zip(idx_knn[j], D[j, idx_knn[j]])]
			heapify(knn_dist)
			txt = "%d,%s\n" % (el[0], [x.obj for x in knn_dist if x.obj!=el[0]])
			outfile.write(txt)
		print "%d slices procesed" % i
		