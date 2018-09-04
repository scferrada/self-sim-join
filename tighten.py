import numpy as np
import sys, os

if len(sys.argv) != 2:
	exit(1)
	
dataset = np.load(sys.argv[1])

vectors = None
l = []
first = True

for row in dataset:
	if first:
		vectors = row
		l.append(row)
		first = False
		continue
	flag = False
	for x in l:	
		if np.array_equal(row, x): 
			flag = True
			break
	if flag: continue
	l.append(row)
	vectors = np.vstack((vectors, row))
	
np.save(open('tightened.data','w'), vectors)