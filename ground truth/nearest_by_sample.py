import argparse, threading, os

parser = argparse.ArgumentParser(description='searches for the nearest neighbor given a set sample')

parser.add_argument('input_csv', type=str, help='The knn csv')
parser.add_argument('output_folder', type=str, help='The directory where the results must be stored')

args = parser.parse_args()

DATA_SIZE = 928276
BATCH_SIZE = DATA_SIZE/10

files = []
for i in range(10):
	files.append(open(os.path.join(args.output_folder,"%d.1nn"%i), 'w'))

for line in open(args.input_csv, 'r'):
	parts = line.split(',')
	source = parts[0].strip()
	for i in range(10):
		found = False
		for part in parts[1:]:
			if part.strip() == '': continue
			target = int(part.strip())
			if target <= (BATCH_SIZE * (i+1)):
				files[i].write("%s,%d\n" % (source, target))
				found = True
				break
		if not found:		
			target = "XXXX"
			files[i].write("%s,%s\n" % (source, target))
for file in files:
	file.close()