import quickjoin as qj
import numpy as np
import sys, os, time

if len(sys.argv) != 3:
	print("usage: python run_qj.py input_matrix ouput_folder")
	exit(1)
	
input_matrix = np.load(sys.argv[1])
output = open(os.path.join(sys.argv[2], "qj.knn"), "w")
output_dists = open(os.path.join(sys.argv[2], "qj.dists"), "w")

print("running QJ for %d vectors" % len(input_matrix))
start = time.time()
res, dist = qj.quickjoin(input_matrix, 10, 1000)
end = time.time()

print("writing results")

for x in results:
	output.write("%d, %s\n" %(x, ",".join([str(w.obj) for w in results[x]])))
	output_dists.write("%d, %s\n" %(x, ",".join([str(w.dist) for w in results[x]])))
	
output.close()
output_dists.close()
print("QJ finished in %.8f seconds, computing %d distances"%(end-start, dist))