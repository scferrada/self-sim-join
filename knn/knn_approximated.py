import numpy as np
import math
from heapq import heappush, heappop, heappushpop

class Res:
    def __init__(self, obj, dist):
        self.dist = -dist
        self.obj = obj

    def __lt__(self, other):
        return self.dist < other.dist

    def __eq__(self, other):
        return self.dist == other.dist and self.obj == other.obj

    def __str__(self):
        return str(self.obj) + "; " + str(self.dist)


class Group:
    def __init__(self, center=None, elems=None, r=0, id=-1):
        self.center = center
        self.elems = elems
        self.id = id
        self.r = r
        self.first = True

    def stack(self, element, dist):
        if self.first:
            self.elems = element
            self.first = False
        else:
            self.elems = np.vstack((self.elems, element))
        if dist > self.r:
            self.r = dist

    def __len__(self):
        if self.center is None: return 0
        if self.elems is None: return 1
        if len(self.elems.shape) == 1: return 2
        return len(self.elems) + 1

    def show(self):
        print(self.all())

    def get_elems(self):
        return self.elems

    def all(self):
        if len(self) == 1: return self.center
        return np.vstack((self.elems, self.center))

    def all_but(self, index):
        if len(self) == 1: return self.center
        if len(self) == 2 and index == 0: return self.center
        return np.vstack((np.vstack((self.elems[:index], self.elems[index + 1:])), self.center))


def get_centers(input_matrix):
    h, w = input_matrix.shape
    idx = np.random.choice(h, size=math.ceil(math.sqrt(h)), replace=False)
    centers = input_matrix[idx, :]
    mask = np.ones(len(input_matrix), dtype=bool)
    mask[idx] = False
    data = input_matrix[mask, :]
    return data, centers


def make_groups(data, centers, k, max_size, results):
    groups = [Group(x, None, 0, id) for id, x in enumerate(centers)]
    for row in data:
        distances = np.sum(np.abs(centers[:, 1:] - row[1:]), axis=1)
        for c, d in enumerate(distances):
            if len(results[centers[c][0]]) < k:
                heappush(results[centers[c][0]], Res(row[0], d))
            elif d < -results[centers[c][0]][0].dist:
                heappushpop(results[centers[c][0]], Res(row[0], d))
        j = 0
        while True:
            idx = np.argpartition(distances, j)[j]
            if len(groups[idx]) < max_size:
                groups[idx].stack(row, distances[idx])
                break
            j += 1
    return groups


def sim_join(input_matrix, k, group_size=1):
    idx = np.arange(len(input_matrix)).reshape(len(input_matrix), 1)
    matrix = np.hstack((idx, input_matrix))
    data, centers = get_centers(matrix)
    results = {x: [] for x in idx}
    print("making %d *sqrt(n)-sized groups..." % group_size)
    groups = make_groups(data, centers, k, group_size * len(centers), results)
    R = np.array([group.r for group in groups])
    print("nested loop... %d groups" % len(groups))
    for i, group in enumerate(groups):
        if len(group) <= 1:
            continue
        if len(group) == 2:
            g = [group.elems]
        else:
            g = group.elems
        for current, elem_i in enumerate(g):
            dist_to_groups = (np.sum(np.abs(centers[:, 1:] - elem_i[1:]), axis=1) - R)
            j = 0
            while True:
                closest_group = np.argpartition(dist_to_groups, j)[j]
                if groups[closest_group].id == group.id:
                    closest_group = np.argpartition(dist_to_groups, j)[j]
                j += 1
            target = np.vstack((group.all_but(current), groups[closest_group].all()))
            while len(target) <= k:
                closest_group = np.argpartition(dist_to_groups, j)[j]
                j += 1
                if groups[closest_group].id == group.id: continue
                target = np.vstack((target, groups[closest_group].all()))
            distances = np.sum(np.abs(target[:, 1:] - elem_i[1:]), axis=1)
            idx = np.argpartition(distances, k)[:k]
            knn = target[idx]
            for d, e in enumerate(knn):
                heappush(results[elem_i[0]], Res(knn[0], distances[d]))
        print("group %d done." % i)
    return results
