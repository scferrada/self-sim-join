from lsh_join import self_sim_join
import numpy as np
import argparse, os

parser = argparse.ArgumentParser(description='Runs the approximated self similarity join (kNN) algorithm of a given ser of points several times')

parser.add_argument('input_matrix', type=str, help='The numpy vector storing file')
parser.add_argument('output_folder', type=str, help='The directory where the results must be stored')
parser.add_argument('--k', dest='k', type=int, default=10, help='The number of nearest neighbors to retrieve. 10 by default.')
parser.add_argument('--r', dest='r', type=int, default=200, help='The number of nearest neighbors to retrieve. 10 by default.')

args = parser.parse_args()

data = np.load(args.input_matrix)

for i in xrange(100):
	results = self_sim_join(data, 10, 30, args.r, args.k)
	with open(os.path.join(args.output_folder, 'lsh%d.res'%i), 'w') as out:
		for x in results:
			out.write("%d, %s\n"%(x, ','.join([str(j) for j in results[x]])))
	print i