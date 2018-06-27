import numpy as np
import math

class Group:
    def __init__(self, center=None, elems=None, r=0):
        self.center=center
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

def make_dict(input_matrix):
    i = 0
    mapping = {}
    for row in input_matrix:
        mapping[np.array2string(row)] = i
        i += 1
    return mapping        
            
def get_centers(input_matrix):
    h, w = input_matrix.shape
    idx = np.random.choice(h, size=math.ceil(math.sqrt(h)), replace=False)
    centers = input_matrix[idx, :]
    mask = np.ones(len(input_matrix), dtype=bool)
    mask[idx] = False
    data = input_matrix[mask,:]
    return data, centers
    
def make_groups(data, centers, k, max_size, results):
    groups = [Group(x, [], 0) for x in centers]
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
    #mapping = make_dict(input_matrix)
    idx = np.arange(len(input_matrix)).reshape(len(input_matrix),1)
    matrix = np.hstack((idx, input_matrix))
    data, centers = get_centers(matrix)
    results = []
    print("making groups...")
    groups = make_groups(data, centers, k, group_size*len(centers), results)
    R = np.array([group.r for group in groups])
    print("nested loop...")
    for i, group in enumerate(groups):
        if len(group) <= 1: continue
        if len(group) == 2: g=[group.elems]
        else: g = group.elems
        for elem_i in g:
            current = 0
            j=1
            dist_to_groups = (np.sum(np.abs(centers[:,1:] - elem_i[1:]), axis=1) - R)
            closest_group = np.argpartition(dist_to_groups, j)[j]
            target = np.vstack((group.all_but(current), groups[closest_group].all()))
            while len(target) < k:
                j+=1
                closest_group = np.argpartition(dist_to_groups, j)[j]
                target = np.vstack((target, groups[closest_group].all()))
            distances = np.sum(np.abs(target[:,1:] - elem_i[1:]), axis=1)
            knn = target[np.argpartition(distances, k+1)[:k+1]]
            knn_str = [x[0] for x in knn]
            results.append((elem_i[0], knn_str))
            current += 1
        print("group %d done." % i)
    return results