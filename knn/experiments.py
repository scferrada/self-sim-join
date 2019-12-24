import knn_approximated as knn
import numpy as np
import argparse, os

parser = argparse.ArgumentParser(description='Runs the approximated self similarity join (kNN) algorithm of a given ser of points several times')

parser.add_argument('input_matrix', type=str, help='The numpy vector storing file')
parser.add_argument('output_folder', type=str, help='The directory where the results must be stored')
#parser.add_argument('--N', dest='iter', type=int, default=1, help='The number of times the experiment must be repeated. 1 by default.')
parser.add_argument('--k', dest='k', type=int, default=1, help='The number of nearest neighbors to retrieve. 1 by default.')

args = parser.parse_args()
data = np.load(args.input_matrix)[:2000000,:]
for c in [1,2,3,10,100,1000]:
	for i in range(10):
		#try:
		print("running %d experiment "%i)
		results = knn.sim_join(data, args.k, c)
		f = open(os.path.join(args.output_folder, str(args.k), str(c), "%d.res"%i), "w")
		for x in results:
			f.write("%d,%s\n"%(x, ",".join([str(y.obj) for y in results[x]])))
		f.close()