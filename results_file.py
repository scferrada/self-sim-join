import os, argparse
import numpy as np
	
parser = argparse.ArgumentParser(description='Computes experiment results')
parser.add_argument('groundtruth_path', type=str, help='the path to the ground truth file')
parser.add_argument('approx_path', type=str, help='the directory where the results are stored')
parser.add_argument('out_path', type=str, help='the directory where the results must be stored')
parser.add_argument('--k', dest='k', type=int, default='1', help='the number of NN considered, 1 by default')

args = parser.parse_args()

ground_truth = {}

#read ground truth:
for line in open(args.groundtruth_path, 'r'):
	line = line.translate(None, '[]') #comment for hog
	parts = line.split(',')
	key = int(float(parts[0].strip()))
	knn = [int(x.strip()) for x in parts[1:]]
	ground_truth[key] = knn

print("Ground truth read")	
	

for pt,ds,fs in os.walk(args.approx_path):
	if len(fs) > 0:
		per_file = []
		for f in fs:
			per_img = 0
			for line in open(os.path.join(pt,f), 'r'):
				if "distances" in line: continue
				correct = 0
				line = line.translate(None, '[]') #comment for hog
				parts = line.split(',')
				key = int(float(parts[0].strip()))
				try:
					ann = [int(float(x.strip())) for x in parts[1:]]
				except:
					continue
				knn = ground_truth[key][:args.k]
				for e in ann:
					if e in knn:
						correct += 1
				per_img+=correct
			per_file.append(per_img)
		print("folder %s completed" % pt)
		out = open(os.path.join(args.out_path, 'c%s_k%d.res'%(pt[pt.rfind("/")+1:],args.k)), 'w')
		c_array = np.array(per_file)
		out.write("%s\n"% np.array2string(c_array))
		out.write("max: %d\n" % np.max(c_array))
		out.write("min: %d\n" % np.min(c_array))
		out.write("avg: %d\n" % np.average(c_array))
		out.write("std: %d\n" % np.std(c_array))
		out.close()
