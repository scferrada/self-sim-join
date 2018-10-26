import knn_approximated as knn
import numpy as np
import argparse, os

parser = argparse.ArgumentParser(description='Runs the approximated self similarity join (1NN) algorithm of a given ser of points several times')

parser.add_argument('input_matrix', type=str, help='The numpy vector storing file')
parser.add_argument('output_folder', type=str, help='The directory where the results must be stored')
parser.add_argument('--N', dest='iter', type=int, default=1, help='The number of times the experiment must be repeated. 1 by default.')
parser.add_argument('--k', dest='k', type=int, default=10, help='The number of nearest neighbors to retrieve. 10 by default.')

args = parser.parse_args()

data = np.load(args.input_matrix)
for i in range(args.iter):
	#try:
	print("running %d experiment"%i)
	results = knn.sim_join(data, args.k, 2)
	f = open(os.path.join(args.output_folder, "%d.res"%i), "w")
	for x in results:
		f.write("%d,%s\n"%(x, ",".join([str(y.obj) for y in results[x]])))
	f.close()
	# except Exception, e:
		# print(str(e))
		# continue