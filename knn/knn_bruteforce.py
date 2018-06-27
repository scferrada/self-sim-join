import argparse, os
import numpy as np 

parser = argparse.ArgumentParser(description='Computes self similarity join (kNN) of a given ser of points')

parser.add_argument('input_numpy', type=str, help='the numpy vectors')
parser.add_argument('output_folder', type=str, help='the directory where the results must be stored')
parser.add_argument('--batch', dest='batch', type=int, default='100', help='the percentage of data to be used')

args = parser.parse_args()

k = 100
d = np.load(args.input_numpy)
dataset = d[:d.shape[0]*(args.batch/100.0)]
print "starting bruteforce for %d" % dataset.shape[0]
with open(os.path.join(args.output_folder, "knn.csv"), "w") as outfile:	
	count = 0
	for row in dataset:
		distances = np.sum(np.abs(dataset - row), axis=1)
		idx_knn = np.argsort(distances)[:k+1]
		txt = "%d,%s\n" % (count, str([x for x in idx_knn.tolist() if x!=count]))
		outfile.write(txt)
		count += 1
		if count % 5000 == 0:
			print "%d vectors procesed" % count