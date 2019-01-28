import numpy as np
import argparse, os

parser = argparse.ArgumentParser(description='Computes experiment histogram')
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
	key = int(parts[0].strip())
	knn = [int(x.strip()) for x in parts[1:]]
	ground_truth[key] = knn

print("Ground truth read")

#compare against results:
for p, d, f in os.walk(args.approx_path):
	if len(f) == 0: 
		continue
	per_img = []
	for file in f:
		path = os.path.join(p, file)
		for line in open(path, 'r'):
			line = line.translate(None, '[]')
			correct = 0
			parts = line.split(',')
			key = int(parts[0].strip())
			ann = [int(float(x.strip())) for x in parts[1:]]
			knn = ground_truth[key][:args.k]
			for e in ann:
				if e in knn:
					correct += 1
			per_img.append(correct)		
	print("%s done, %d files"%(p, len(f)))
	np.save(os.path.join(args.out_path, 'histogram%s.npy'%p[-1]), np.array(per_img))