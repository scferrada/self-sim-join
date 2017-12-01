import argparse, threading
import quickjoin as qj
import numpy as np

parser = argparse.ArgumentParser(description='Runs sampled experiments to measure the performance of Quickjoin')

parser.add_argument('input_matrix', type=str, help='The path to the numpy matrix')
parser.add_argument('output_path', type=str, help='The path where the output must be stored')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=100, help='The size of the sample incrementation, as a percentage. Default 100')
parser.add_argument('--N', dest='iter', type=int, default=1, help='The number of times the experiment must be repeated. 1 by default.')
parser.add_argument('--i', dest='i', type=int, default=9, help='The number of the batch. 9 by default.')

args = parser.parse_args()

DATA_SIZE = 928276
MAX_THREADS = 1#0

hashing = {}

data = np.load(args.input_matrix)

count = 0
for row in data:
	hashing[row.tostring()] = count
	count += 1

def run_t(id):	
	for j in range(args.iter/MAX_THREADS):
		#for i in range(100/args.batch_size):
		print("Running %d iteration, %d batch, thread %d" % (j+1, args.i+1, id))
		results = {}
		total_dists = 0
		until = (DATA_SIZE/10)*(1+args.i)
		res, distances = qj.quickjoin(data[:until], 0, 10, 10, 0.5)
		results.update(res)
		total_dists += distances
		res, distances = qj.quickjoin(data[:until], 35, 10, 10, 0.5)
		results.update(res)
		total_dists += distances
		outfile = open(os.path.join(args.outfile, "%d_%d_%d" % (id,args.i+1,j+1)),'w')
		for (source, target) in res:
			txt = "%s; %s\n"
			source_id = hashing[source]
			target_id = hashing[target]
			outfile.write(txt % (source_id, target_id))
		outfile.write("%d distances computed" % total_dists)
		outfile.close()

threads = []
for i in range(MAX_THREADS):
	t = threading.Thread(target=run_t, args=(i, ))
	threads.append(t)
	t.start()
	
for t in threads:
	t.join()
