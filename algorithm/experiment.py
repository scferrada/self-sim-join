import argparse, threading
import approximated as ap

parser = argparse.ArgumentParser(description='Runs the approximated self similarity join (1NN) algorithm of a given ser of points several times')

parser.add_argument('input_matrix', type=str, help='The numpy vector storing file')
parser.add_argument('output_folder', type=str, help='The directory where the results must be stored')
parser.add_argument('--N', dest='iter', type=int, default=1, help='The number of times the experiment must be repeated. 1 by default.')
parser.add_argument('--size', dest='factor', type=int, default=1, help='The factor of the group size. 1 by default.')

args = parser.parse_args()

MAX_THREADS = 20

def run_t(id):
	for count in range(args.iter/MAX_THREADS):
		print("Iteration %d of %d. Thread %d" % (count, args.iter/MAX_THREADS, id))
		ap.sim_join(args.input_matrix, args.output_folder, factor=args.factor, iteration="%d_%d"%(id, count))

threads = []
for i in range(MAX_THREADS):
	t = threading.Thread(target=run_t, args=(i, ))
	threads.append(t)
	t.start()
	
for t in threads:
	t.join()