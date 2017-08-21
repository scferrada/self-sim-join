import argparse, os
import numpy as np 

parser = argparse.ArgumentParser(description='Computes self similarity join (1NN) of a given ser of points')

parser.add_argument('input_folder', type=str, help='the directory where the numpy vectors are')
parser.add_argument('output_folder', type=str, help='the directory where the results must be stored')
parser.add_argument('--ext', dest='extension', type=str, default='npy', help='the extension of the numpy files, .npy by default')

args = parser.parse_args()

k = 100
print "loading matrices"
matrices = []
for (p,d,f) in os.walk(args.input_folder):
	matrices.extend([os.path.join(p,x) for x in f if x.endswith(args.extension)])

first = True
dataset = None
for m in matrices:
	if first:
		dataset = np.load(m)
		first = False
	else:
		dataset = np.vstack((dataset, np.load(m)))
print "%d matrices loaded" % dataset.shape[0]
np.save(open('dataset.npy', 'w'), dataset)
print "starting bruteforce"
with open(os.path.join(args.output_folder, "knn.csv"), "w") as outfile:	
	count = 0
	for row in dataset:
		distances = np.abs(np.sum(dataset - row, axis=1))
		idx = np.argsort(distances)[:k]
		idx_str = np.array2string(idx, separator=',')[2:-2].strip()
		txt = "%d,%s\n" % (count, idx_str)
		outfile.write(txt)
		count += 1
		if count % 5000 == 0:
			print "%d vectors procesed" % count