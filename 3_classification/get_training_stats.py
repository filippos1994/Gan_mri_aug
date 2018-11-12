import numpy as np

from PIL import Image
from os import path, walk

train_dir = "train"
img_list = []
counter = 0
for root, _, files in walk(train_dir):
	for file in files:
		counter += 1
		if (counter % 100 == 0):
			print(counter)
		img = Image.open(path.join(root, file))

		img = np.asarray(img)
		img_list.append(img)


print(len(img_list))
array = np.asarray(img_list)


mean = np.mean(array)
std = np.std(array)

print(mean, std)
# print(type(mean), type(std))
