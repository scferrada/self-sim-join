from pyflann import *
import numpy as np
from datetime import datetime


dataset = np.load('../dataset.npy')
print dataset.shape
flann = FLANN()
print "Finding nearest neighbors"
start = datetime.now()
params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9);
print params
result, dists = flann.nn_index(dataset,10, checks=params["checks"]);
end = datetime.now()
total_time = (end - start).total_seconds()
print "neighbors found in %d" % total_time
print "writing results"
try:
	out = open("time.txt","w")
	out.write("%d seconds for %d images\n" % (total_time, dataset.shape[0]))
	out.close()
except:
	print "couldn't write file, time was %d seconds" % total_time
np.save(open("result.txt", "w"), result)
np.save(open("dists.txt", "w"), dists)