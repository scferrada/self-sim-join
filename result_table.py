import argparse, os, math

parser = argparse.ArgumentParser(description='Measures the precision of the approximated 1-NN join algorithm of Bustos et al.')

parser.add_argument('ground_truth', type=str, help='The file with the actual 100-NN of each vector')
parser.add_argument('approx_path', type=str, help='The folder with the results of the algorithm execution')
parser.add_argument('output_path', type=str, help='The folder where the output must be stored')

args = parser.parse_args()

def mean(elems):
	return sum(elems)/len(elems)
	
def stdv(elems):
	m = mean(elems)
	s = 0
	for el in elems:
		s += (el-m) * (el-m)
	return math.sqrt(s/len(elems))

print "reading ground truth"
data_size = 0
ground_truth = {}
for line in open(args.ground_truth, "r"):
	parts = line.split(',')
	source = int(parts[0].strip())
	if source in ground_truth:
		print "%d already in dict" % source
	ground_truth[source] = []
	for target in parts[1:]:
		if target == source:
			continue
		try:
			ground_truth[source].append(int(target.strip()))
		except:
			continue
	if len(ground_truth[source]) < 99:
		print "less than 100NN for %d: %d" % (source, len(ground_truth[source]))
	data_size += 1
	
result_files = []
for p, d, f in os.walk(args.approx_path):
	result_files.extend(f)
	break
	
correct_1nn = [0 for x in result_files]
on_10nn = [0 for x in result_files]
on_20nn = [0 for x in result_files]
on_30nn = [0 for x in result_files]
on_40nn = [0 for x in result_files]
on_50nn = [0 for x in result_files]
on_60nn = [0 for x in result_files]
on_70nn = [0 for x in result_files]
on_80nn = [0 for x in result_files]
on_90nn = [0 for x in result_files]
on_100nn = [0 for x in result_files]
beyond_100nn = [0 for x in result_files]

print "analyzing"
current_file = 0
rescount = 0
for file in result_files:
	for line in open(os.path.join(args.approx_path, file), 'r'):
		rescount += 1
		if line.startswith('dist'): continue
		parts = line.split(',')
		source = int(parts[0].strip())
		target = int(parts[1].strip())
		try:
			knn = ground_truth[source]
			index = knn.index(target)
			if index <= 1:
				correct_1nn[current_file] += 1
				continue
			if index < 10:
				on_10nn[current_file] += 1
				continue
			if index < 20:
				on_20nn[current_file] += 1
				continue
			if index < 30:
				on_30nn[current_file] += 1
				continue
			if index < 40:
				on_40nn[current_file] += 1
				continue
			if index < 50:
				on_50nn[current_file] += 1
				continue
			if index < 60:
				on_60nn[current_file] += 1
				continue
			if index < 70:
				on_70nn[current_file] += 1
				continue
			if index < 80:
				on_80nn[current_file] += 1
				continue
			if index < 90:
				on_90nn[current_file] += 1
				continue
			if index < 100:
				on_100nn[current_file] += 1
				continue
		except ValueError:
			beyond_100nn[current_file] += 1
		except KeyError:
			print "no key found for %d" % source
			exit()
	if current_file%10 == 0:
		print "%d files analyzed" % current_file
	current_file += 1
	print rescount
	rescount = 0

means = []
means.append(mean(correct_1nn))
means.append(mean(on_10nn))
means.append(mean(on_20nn))
means.append(mean(on_30nn))
means.append(mean(on_40nn))
means.append(mean(on_50nn))
means.append(mean(on_60nn))
means.append(mean(on_70nn))
means.append(mean(on_80nn))
means.append(mean(on_90nn))
means.append(mean(on_100nn))
means.append(mean(beyond_100nn))	

stdvs = []
stdvs.append(stdv(correct_1nn))
stdvs.append(stdv(on_10nn))
stdvs.append(stdv(on_20nn))
stdvs.append(stdv(on_30nn))
stdvs.append(stdv(on_40nn))
stdvs.append(stdv(on_50nn))
stdvs.append(stdv(on_60nn))
stdvs.append(stdv(on_70nn))
stdvs.append(stdv(on_80nn))
stdvs.append(stdv(on_90nn))
stdvs.append(stdv(on_100nn))
stdvs.append(stdv(beyond_100nn))

mins = []
mins.append(min(correct_1nn))
mins.append(min(on_10nn))
mins.append(min(on_20nn))
mins.append(min(on_30nn))
mins.append(min(on_40nn))
mins.append(min(on_50nn))
mins.append(min(on_60nn))
mins.append(min(on_70nn))
mins.append(min(on_80nn))
mins.append(min(on_90nn))
mins.append(min(on_100nn))
mins.append(min(beyond_100nn))

maxs = []
maxs.append(max(correct_1nn))
maxs.append(max(on_10nn))
maxs.append(max(on_20nn))
maxs.append(max(on_30nn))
maxs.append(max(on_40nn))
maxs.append(max(on_50nn))
maxs.append(max(on_60nn))
maxs.append(max(on_70nn))
maxs.append(max(on_80nn))
maxs.append(max(on_90nn))
maxs.append(max(on_100nn))
maxs.append(max(beyond_100nn))

with open(os.path.join(args.output_path, 'results.txt'), 'w') as outfile:
	outfile.write('means: %s\n' % ', '.join([str(x) for x in means]))
	outfile.write('stdvs: %s\n' % ', '.join([str(x) for x in stdvs]))
	outfile.write('mins: %s\n' % ', '.join( [str(x) for x in mins ]))
	outfile.write('maxs: %s\n' % ', '.join( [str(x) for x in maxs ]))
	outfile.write('total points: %d' % data_size)
	