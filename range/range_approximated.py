import numpy as np
import math

class Group:
    def __init__(self, center=None, elems=None, r=0, id=-1):
        self.center=center
        self.elems=elems
        self.id = id
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
        if self.elems is None: return 1
        if len(self.elems.shape) == 1: return 2
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
        if len(self)==2 and index == 0: return self.center
        return np.vstack((np.vstack((self.elems[:index], self.elems[index+1:])), self.center))       
            
def get_centers(input_matrix):
    h, w = input_matrix.shape
    idx = np.random.choice(h, size=math.ceil(math.sqrt(h)), replace=False)
    centers = input_matrix[idx, :]
    mask = np.ones(len(input_matrix), dtype=bool)
    mask[idx] = False
    data = input_matrix[mask,:]
    return data, centers
    
def make_groups(data, centers, r, max_size, results):
    groups = [Group(x, None, 0, id) for id, x in enumerate(centers)]
    center_r = [[] for x in centers]
    for row in data:
        distances = np.sum(np.abs(centers[:,1:]-row[1:]), axis=1)
        idr = np.argwhere(distances <= r)
        for i in idr:
            center_r[i].append(row[0])
        indices = np.argsort(distances)
        for index in indices:
            if len(groups[index]) < max_size:
                groups[index].stack(row, distances[index])
                break
    for j, center in enumerate(centers):
        results.append((center[0], center_r[j]))
    return groups

def sim_join(input_matrix, r, group_size=1):
    idx = np.arange(len(input_matrix)).reshape(len(input_matrix),1)
    matrix = np.hstack((idx, input_matrix))
    data, centers = get_centers(matrix)
    results = []
    print("making %d *sqrt(n)-sized groups..."%group_size)
    groups = make_groups(data, centers, r, group_size*len(centers), results)
    R = np.array([group.r for group in groups])
    print("nested loop... %d groups" %len(groups))
    for i, group in enumerate(groups):
        print len(group)
        if len(group) <= 1: continue
        if len(group) == 2: g=[group.elems]
        else: g = group.elems
        for current, elem_i in enumerate(g):
            dist_to_groups = (np.sum(np.abs(centers[:,1:] - elem_i[1:]), axis=1) - R)
            closest_group = np.argmin(dist_to_groups)
            if groups[closest_group].id == group.id: 
                closest_group = np.argpartition(dist_to_groups, 1)[1]
            target = np.vstack((group.all_but(current), groups[closest_group].all()))
            distances = np.sum(np.abs(target[:,1:] - elem_i[1:]), axis=1)
            matches = target[distances <=r]
            results.append((elem_i[0], [x[0] for x in matches]))
        print("group %d done." % i)
    return results