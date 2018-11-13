# After the images of each dataset have been decided
# we run this script to copy them

from distutils.dir_util import copy_tree
from shutil import copy2

data_dir  = 'data_gan_adnorfirst/'
gan_dir   = data_dir + 'gan/'
train_src = data_dir + 'train/'

for i in range(1,9):
	print(i)
	ratio = i * 25
	if (ratio < 100):
		ratio_str = '0' + str(ratio)
	else:
		ratio_str = str(ratio)


	train_dst = data_dir + 'train_' + ratio_str + '/'
	copy_tree(train_src, train_dst)

	for TYPE in ['ad','nor']:
		gan_src = gan_dir + TYPE + '/'

		fake_path = 'fake_names/fake_imgs_{}_{}.txt'.format(TYPE, ratio_str)
		fake_file = open(fake_path, 'r')

		for fake_img in fake_file:
			src = gan_src + fake_img.split('\n')[0]
			dst = train_dst + TYPE + '/'
			copy2(src,dst)
