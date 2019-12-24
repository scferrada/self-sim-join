import argparse, threading, os
import quickjoin as qj
import numpy as np

parser = argparse.ArgumentParser(description='Runs sampled experiments to measure the performance of Quickjoin')

parser.add_argument('input_matrix', type=str, help='The path to the numpy matrix')
parser.add_argument('output_path', type=str, help='The path where the output must be stored')
parser.add_argument('--k', dest='k', type=int, default=1, help='The number of neares neighbors required.')
parser.add_argument('--N', dest='iter', type=int, default=1, help='The number of times the experiment must be repeated. 1 by default.')

args = parser.parse_args()

MAX_THREADS = 5

data = np.load(args.input_matrix)
c = len(data)//1000 #.1% of the data
def run_t(id):	
	for j in range(args.iter//MAX_THREADS):
		print("Running %d iteration, thread %d" % (j+1, id))
		res, distances = qj.quickjoin(data, args.k, c)
		outfile = open(os.path.join(args.output_path, "%d_%d_%d" % (id,j+1,args.k)),'w')
		for idx in res:
			txt = "%s; %s\n"
			outfile.write(txt % (idx, ';'.join([str(x.obj) for x in res[idx]])))
		outfile.write("%d distances computed" % distances)
		outfile.close()

threads = []
for i in range(MAX_THREADS):
	t = threading.Thread(target=run_t, args=(i, ))
	threads.append(t)
	t.start()
	
for t in threads:
	t.join()
