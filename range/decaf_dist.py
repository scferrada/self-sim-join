import numpy as np
import os, random

files = []
preffix = '../data/decaf7'
for p,d,f in os.walk(preffix):
	files.extend(f)
	break

dists = []	
for _ in range(1000):
	candidates = random.sample(files, 2)
	dists.append(np.sum(np.abs(np.load(os.path.join(preffix,candidates[0]))-np.load(os.path.join(preffix,candidates[1])))))
	
print("average dist: %6f"%(sum(dists)/float(len(dists))))
print("min dist: %6f" % min(dists))
print("max dist: %6f" % max(dists))
print("1st quartile: %6f" %np.percentile(np.array(dists), 25))