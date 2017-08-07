import argparse, os
import numpy as np 

parser = argparse.ArgumentParser(description='Computes self similarity join (1NN) of a given ser of points')

parser.add_argument('input_folder', type=str, help='the directory where the numpy vectors are')
parser.add_argument('output_folder', type=str, help='the directory where the results must be stored')
parser.add_argument('--ext', dest='extension', type=str, default='npy', help='the extension of the numpy files, .npy by default')

args = parser.parse_args()

k = 100

matrices = []
for (p,d,f) in os.walk(args.input_folder):
	matrices.extend([x for x in f if f.endswith(args.extension)])

first = True
dataset = None
for m in matrices:
	if first:
		dataset = m
		first = False
	else:
		dataset = np.vstack(dataset, m)
with open(os.path.join(args.output_folder, "knn.csv"), "w") as outfile:	
	count = 0
	for row in dataset:
		distances = np.abs(dataset - row)
		idx = np.argsort(distances)[:k]
		idx_str = np.array2string(idx, separator=',')[2:-2].strip()
		txt = "%d,%s\n" % (count, idx_str)
		outfile.write(txt)