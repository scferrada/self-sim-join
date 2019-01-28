import math, random
import numpy as np
from pyjarowinkler import distance


hash = {}

def make_hash(input_array):
    for i, s in enumerate(input_array):
        hash[s] = i

class Group:
    def __init__(self, center=None, elems=[], r=0, id=-1):
        self.center=center
        self.elems=elems
        self.id = id
        self.r=r

    def stack(self, element, dist):
        self.elems.append(element)
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
        if len(self)==1: return [self.center]
        return self.elems + [self.center]
        
    def all_but(self, index):
        if len(self)==1: return self.center
        if index+1 < len(self):
            return [self.center] + self.elems[:index] + self.elems[index+1:]
        else:
            return [self.center] + self.elems[:index]
            
def get_centers(input_array):
    copy = list(input_array)
    random.shuffle(copy)
    return copy[int(math.ceil(math.sqrt(len(input_array)))):], copy[:int(math.ceil(math.sqrt(len(input_array))))]
    
def make_groups(data, centers, k, max_size, results):
    groups = [Group(x, [], 0, id) for id, x in enumerate(centers)]
    center_nn = [[-1 for _ in range(k)] for _ in range(len(centers))]
    center_mindists = [[float("inf") for _ in range(k)] for _ in range(len(centers))]
    for row in data:
        dists = [distance.get_jaro_distance(c, row) for c in centers]
        for i, c in enumerate(centers):
            if dists[i] < max(center_mindists[i]):
                idx = center_mindists[i].index(max(center_mindists[i]))
                center_nn[i][idx] = hash[row]			
                center_mindists[i][idx] = dists[i]
        best_groups = np.argsort(np.array(dists)).tolist()
        t = 0
        while True:
            if len(groups[best_groups[t]]) < max_size:
                groups[best_groups[t]].stack(row, dists[best_groups[t]])
                break
            t += 1
    for i, center in enumerate(centers):
        results.append((hash[center], center_nn[i]))
        #results.append((center, center_nn[i]))
    return groups

def sim_join(input_array, k, group_size=1):
    make_hash(input_array)
    data, centers = get_centers(input_array)
    results = []
    print("making %d *sqrt(n)-sized groups..."%group_size)
    groups = make_groups(data, centers, k, group_size*len(centers), results)
    print("nested loop %d groups" % len(groups))
    for i, group in enumerate(groups):
        if len(group) <= 1: continue
        for current, elem_i in enumerate(group.elems):
            j=1
            dist_to_groups = [distance.get_jaro_distance(g.center, elem_i)-g.r for g in groups]
            closest_group = dist_to_groups.index(min(dist_to_groups))
            if groups[closest_group].id == group.id: 
                closest_group = np.argpartition(np.array(dist_to_groups), j)[j]
                j+=1
            target = group.all_but(current) + groups[closest_group].all()
            while len(target) <= k:
                closest_group = np.argpartition(np.array(dist_to_groups), j)[j]
                j+=1
                if groups[closest_group].id == group.id: continue
                target = target + groups[closest_group].all()
            distances = np.array([distance.get_jaro_distance(x, elem_i) for x in target])
            knn = np.argpartition(distances, k)[:k].tolist()
            results.append((hash[elem_i], [hash[target[x]] for x in knn]))
    return results