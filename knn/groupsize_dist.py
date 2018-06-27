from knn_approximated import Group, get_centers
import numpy as np
import argparse, os

parser = argparse.ArgumentParser(description='Computes the make groups routine 100 times and obtains a distribution of the group sizes')

parser.add_argument('input_matrix', type=str, help='The numpy vector storing file')
parser.add_argument('output_folder', type=str, help='The directory where the results must be stored')

args = parser.parse_args()

	
def make_groups(data, centers, max_size):
    groups = [Group(x, [], 0) for x in centers]
    for row in data:
        distances = np.sum(np.abs(centers-row), axis=1)
        indices = np.argsort(distances)
        for index in indices:
            if len(groups[index]) < max_size:
                groups[index].stack(row, distances[index])
                break
    return groups

matrix = np.load(args.input_matrix)
output = open(os.path.join(args.output_folder, 'dist.txt'), 'w')
for i in xrange(100):
    print i
    data, centers = get_centers(matrix)
    groups = make_groups(data, centers, 2*len(centers))
    for group in groups:
        output.write("%d\n"%len(group))
output.close()