import numpy as np
import argparse, os
from pyflann import *

parser = argparse.ArgumentParser(description='Runs the approximated knn self similarity join algorithm of a given ser of points 100 several times')

parser.add_argument('input_matrix', type=str, help='The numpy vector storing file')
parser.add_argument('output_folder', type=str, help='The directory where the results must be stored')
parser.add_argument('--k', dest='k', type=int, default=10, help='The number of nearest neighbors to retrieve. 10 by default.')

args = parser.parse_args()
data = np.load(args.input_matrix)

for i in xrange(100):
	findex = FLANN()
	set_distance_type('manhattan')
	params = findex.build_index(data, algorithm="autotuned", target_precision=0.9)
	result, dists = findex.nn_index(data, args.k, checks=params["checks"]);
	np.save(open(os.path.join(args.output_folder,"%d.res"%i), "w"), result)
	print("%d iteration completed"%i)