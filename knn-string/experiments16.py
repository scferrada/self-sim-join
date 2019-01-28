import knn_approximated as knn
import numpy as np
import argparse, os

parser = argparse.ArgumentParser(description='Runs the approximated self similarity join (1NN) algorithm of a given ser of points several times')

parser.add_argument('input_matrix', type=str, help='The numpy vector storing file')
parser.add_argument('output_folder', type=str, help='The directory where the results must be stored')

args = parser.parse_args()

data = []
for line in open(args.input_matrix, "r"):
	data.append(line.strip())

k=16
for c in [1, 2, 3]:
	for i in range(100):
		print("running %d experiment"%i)
		results = knn.sim_join(data, k, c)
		f = open(os.path.join(args.output_folder, str(k), str(c), "%d.res"%i), "w")
		for x, nn in results:
			f.write("%d,%s\n"%(x, str(nn)))
		f.close()