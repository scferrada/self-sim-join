import numpy as np
import os, argparse

parser = argparse.ArgumentParser(description='Computes experiment results')
parser.add_argument('groundtruth_path', type=str, help='the path to the ground truth file')
parser.add_argument('approx_path', type=str, help='the directory where the results are stored')
parser.add_argument('out_path', type=str, help='the directory where the results must be stored')
parser.add_argument('--k', dest='k', type=int, default='1', help='the number of NN considered, 1 by default')

args = parser.parse_args()

print "reading GT"
ground_truth = {}
for line in open(args.groundtruth_path, 'r'):
	line = line.translate(None, '[]')
	parts = line.split(',')
	key = int(parts[0].strip())
	knn = [int(x.strip()) for x in parts[1:]]
	ground_truth[key] = knn

print "GT read"
for p,d,f in os.walk(args.approx_path):
	if len(f) == 0: 
		continue
	correct_per_file = []
	for file in f:
		res = np.load(os.path.join(args.approx_path, file))
		correct = 0
		for i, v in enumerate(res):
			knn = ground_truth[i][:args.k]
			for e in v:
				if e in knn:
					correct += 1
		correct_per_file.append(correct)
		print correct
	out = open(os.path.join(args.out_path, "k%d"%(args.k)), 'w')
	c_array = np.array(correct_per_file)
	out.write("%s\n"% np.array2string(c_array))
	out.write("max: %d\n" % np.max(c_array))
	out.write("min: %d\n" % np.min(c_array))
	out.write("avg: %d\n" % np.average(c_array))
	out.write("std: %d\n" % np.std(c_array))
	out.close()
		