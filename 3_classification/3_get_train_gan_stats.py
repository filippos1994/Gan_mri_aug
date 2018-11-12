import numpy as np

from PIL import Image
from os import path, walk

for i in range(6,9):
	ratio = i*25
	if ratio < 100:
		ratio_str = '0' + str(ratio)
	else:
		ratio_str = str(ratio)

	print(ratio_str)
	train_dir = 'train_' + ratio_str
	img_list = []
	counter = 0
	for root, _, files in walk(train_dir):
		for my_file in files:
			counter += 1
			if (counter % 30000 == 0):
				print(counter)
			img = Image.open(path.join(root, my_file))

			img = np.asarray(img)
			img_list.append(img)


	print(len(img_list))
	array = np.asarray(img_list)


	mean = np.mean(array)
	std = np.std(array)

	print(mean, std)

	stats_path = 'stats_train/stats_{}.txt'.format(ratio_str)
	try:
		stats_file = open(stats_path, 'w')
	except IOError:
		# If not exists, create the file
		stats_file = open(stats_path, 'w+')

	stats_file.write(str(mean) + '\n')
	stats_file.write(str(std) + '\n')

	stats_file.close()
# print(type(mean), type(std))