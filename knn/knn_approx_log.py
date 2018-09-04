import numpy as np
import math

class Group:
    def __init__(self, center=None, elems=None, id=-1, r=0):
        self.center=center
        self.id = id
        self.elems=elems
        self.r=r
        self.first = True

    def stack(self, element, dist):
        if self.first:
            self.elems = element
            self.first = False
        else:
            self.elems = np.vstack((self.elems, element))
        if dist>self.r: 
            self.r=dist
            
    def __len__(self):
        if self.center is None: return 0
        return len(self.elems) + 1
    
    def show(self):
        print(self.all())
        
    def get_elems(self):
        return self.elems

    def all(self):
        if len(self)==1: return self.center
        return np.vstack((self.elems, self.center))
        
    def all_but(self, index):
        if len(self)==1: return self.center
        return np.vstack((np.vstack((self.elems[:index], self.elems[index+1:])), self.center))   
            
def get_centers(input_matrix):
    h, w = input_matrix.shape
    idx = np.random.choice(h, size=math.ceil(math.sqrt(h)), replace=False)
    centers = input_matrix[idx, :]
    mask = np.ones(len(input_matrix), dtype=bool)
    mask[idx] = False
    data = input_matrix[mask,:]
    return data, centers
    
def make_groups(data, centers, k, max_size, results):
    groups = [Group(x, [], 0, id) for id, x in enumerate(centers)]
    center_nn = np.full((len(centers),k), -1)
    center_mindists = np.full((len(centers),k), np.inf)
    for row in data:
        distances = np.sum(np.abs(centers[:,1:]-row[1:]), axis=1)
        max_min_idx = np.argmax(center_mindists, axis=1)
        id0 = np.indices(max_min_idx.shape)
        lower_idx = distances < center_mindists[id0, max_min_idx]
        temp = center_mindists[id0, max_min_idx]
        temp[lower_idx] = distances[lower_idx[0]]
        center_mindists[id0, max_min_idx] = temp
        temp = center_nn[id0, max_min_idx]
        temp[lower_idx] = row[0]
        center_nn[id0, max_min_idx] = temp
        indices = np.argsort(distances)
        for index in indices:
            if len(groups[index]) < max_size:
                groups[index].stack(row, distances[index])
                break
    j=0
    for center in centers:
        results.append((center[0], center_nn[j].tolist()))
        j+=1
    return groups

def sim_join(input_matrix, k, group_size=1):
    idx = np.arange(len(input_matrix)).reshape(len(input_matrix),1)
    matrix = np.hstack((idx, input_matrix))
    data, centers = get_centers(matrix)
    results = []
    groups_visited = []
    print("making groups...")
    groups = make_groups(data, centers, k, group_size*len(centers), results)
    R = np.array([group.r for group in groups])
    print("nested loop... for %d groups" % len(groups))
    for i, group in enumerate(groups):
        if len(group) <= 1: continue
        if len(group) == 2: g=[group.elems]
        else: g = group.elems
        for elem_i in g:
            current = 0
            target = group.all_but(current)
            dist_to_groups = (np.sum(np.abs(centers[:,1:] - elem_i[1:]), axis=1) - R)
            sorted_groups = np.argsort(dist_to_groups)
            gidx = 0
            while target.shape[0] <= k:
                if groups[sorted_groups[gidx]].id == group.id:
                    gidx +=1
                target = np.vstack((target, groups[sorted_groups[gidx]].all()))
                gidx +=1
            distances = np.sum(np.abs(target[:,1:] - elem_i[1:]), axis=1)
            candidates = np.sort(np.partition(distances, k)[:k]).tolist()
            knn = candidates
            while gidx < len(groups):
                if max(knn) < dist_to_groups[gidx]:
                    break
                if groups[sorted_groups[gidx]].id == group.id:
                    gidx +=1
                    if gidx >= len(groups): break
                target = groups[sorted_groups[gidx]].all()
                distances = np.sum(np.abs(target[:,1:] - elem_i[1:]), axis=1)
                candidates = distances[distances < max(knn)]
                for c in candidates:
                    max_index, max_value = max(enumerate(knn), key=lambda p: p[1])
                    if c < max_value:
                        knn[max_index] = c
                gidx+=1
            groups_visited.append(gidx)
            results.append((elem_i[0], knn))
            current += 1
        print("group %d done." % i)
    return results, groups_visited