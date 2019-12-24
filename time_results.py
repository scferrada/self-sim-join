import os, argparse
import numpy as np

parser = argparse.ArgumentParser(description='get te average time that the experiments took')

parser.add_argument('input_path', type=str, help='the folder with the experiment results')
parser.add_argument('output_path', type=str, help='the directory where the results must be stored')
parser.add_argument('--k', dest='k', type=int, default='1', help='the number of NN considered, 1 by default')

args = parser.parse_args()

for p, d, f in os.walk(args.input_path):
	if len(f) == 0: 
		continue
	start = 0
	diffs = []
	sorted_files = sorted(f, key=lambda x: os.path.getmtime(os.path.join(p,x))) 
	for file in sorted_files:
		if start == 0:
			start = os.path.getmtime(os.path.join(p, file))
			continue
		end = os.path.getmtime(os.path.join(p, file))
		diffs.append(end-start)
		start = end
	out = open(os.path.join(args.output_path, "c%s_k%d"%(p[p.rfind('/')+1:], args.k)), 'w')
	c_array = np.array(diffs)
	c_array = np.sort(c_array)
	out.write("[%s]\n"% ','.join([str(x) for x in diffs]))
	out.write("max: %d\n" % np.max(c_array))
	out.write("min: %d\n" % np.min(c_array))
	out.write("avg: %d\n" % np.average(c_array))
	out.write("std: %d\n" % np.std(c_array))
	out.close()