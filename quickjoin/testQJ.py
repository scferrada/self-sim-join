import numpy as np
import quickjoin as qj

N = 1000
l = []
for i in range(N):
    l.append([i, i])

data = np.array(l)

res, _ = qj.quickjoin(data, 2, 40)

# for i,x in enumerate(res):
# print i, res[x][0].obj, res[x][1].obj

assert 1 in [x.obj for x in res[0]]
assert 2 in [x.obj for x in res[0]]
assert N - 2 in [x.obj for x in res[N - 1]]
assert N - 3 in [x.obj for x in res[N - 1]]

for i in range(1, N - 1):
    print i
    r = [x.obj for x in res[i]]
    print r
    assert i - 1 in r
    assert i + 1 in r
