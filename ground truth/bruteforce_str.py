import argparse, os
import numpy as np 
from pyjarowinkler import distance

parser = argparse.ArgumentParser(description='Computes self similarity join of a given ser of strings')

parser.add_argument('input', type=str, help='the words')
parser.add_argument('output_folder', type=str, help='the directory where the results must be stored')

args = parser.parse_args()

k = 20

dataset = []
for line in open(args.input, 'r'):
	dataset.append(line.strip())
print "starting bruteforce"
with open(os.path.join(args.output_folder, "strknnj.csv"), "w") as outfile:	
	count = 0
	for row in dataset:
		distances = np.array([distance.get_jaro_distance(x, row) for x in dataset])
		idx = np.argsort(distances)[:k+1]
		idx_str = ",".join([str(x) for x in idx.tolist() if dataset[x]!=row])
		txt = "%d,%s\n" % (count, idx_str)
		outfile.write(txt)
		count += 1
		if count % 500 == 0:
			print "%d words procesed" % count