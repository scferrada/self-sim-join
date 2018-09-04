import argparse, os, editdistance
import numpy as np 

parser = argparse.ArgumentParser(description='Computes self similarity join (kNN) of a given ser of points')

parser.add_argument('input_txt', type=str, help='the word dataset, must be a file with one word per line')
parser.add_argument('output_folder', type=str, help='the directory where the results must be stored')

args = parser.parse_args()

k = 50
dataset = []
hash = {}
for line in open(args.input_txt):
	dataset.append(line.strip())
dataset = dataset
print "starting bruteforce for %d" % len(dataset)
with open(os.path.join(args.output_folder, "knn.csv"), "w") as outfile:	
	count = 0
	for row in dataset:
		distances = np.array([int(editdistance.eval(x, row)) for x in dataset])
		idx_knn = np.argsort(distances, kind='mergesort')[:k]
		txt = "%s,%s\n" % (count, str([x for x in idx_knn.tolist() if dataset[x]!=row]))
		outfile.write(txt)
		count += 1
		if count % 100 == 0:
			print "%d vectors procesed" % count