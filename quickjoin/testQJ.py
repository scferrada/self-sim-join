import numpy as np
import quickjoin as qj

N = 100
l = []
for i in range(N):
	l.append([i,i])
	
data = np.array(l)

res = qj.quickjoin(data, 2, 4)

print len(res)
print len(res[0])
print res[1]

print [x.obj for x in res[0][30]]

#assert 2 in [x.obj for x in res[0]]
#assert 3 in [x.obj for x in res[0]]
#assert N-1 in [x.obj for x in res[N]]
#assert N-2 in [x.obj for x in res[N]]

#for i in range(1,N-1):
#	r = [res[i][x].obj for x in res[i]]
#	assert i-1 in r
#	assert i+1 in r
	