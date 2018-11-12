# We experimented with eight fake to real ratios: 25%, 50%,..., 175%, 200%.
# In order to not have the same images across the different training sets,
# we create 50k images per class and shuffle them 8 times,
# each time only keeping the 12.5% (25% fake to real), 25% (50%),
# ..., 100% (200% fake to real). The resulting images are saved on .txt files,
# so they could be retrieved on another machine.

from os import listdir
from random import shuffle


def get_perms(gan_dir, TYPE):
	ad_lst = listdir(gan_dir + TYPE)
	print(len(ad_lst))
	smallest = len(ad_lst)//8

	for i in [1,2,3,4,5,6,7,8]:
		shuffle(ad_lst)
		ad_lst_ratio = ad_lst[:(smallest*i)]
		print(len(ad_lst_ratio))

		if i < 4:
			fake_path = 'fake_names/fake_imgs_{}_0{}.txt'.format(TYPE, i*25)
		else:
			fake_path = 'fake_names/fake_imgs_{}_{}.txt'.format(TYPE, i*25)

		try:
			fake_file = open(fake_path, 'w')
		except IOError:
			# If not exists, create the file
			fake_file = open(fake_path, 'w+')
	
		for name in ad_lst_ratio:
			fake_file.write(name + '\n')
	
		fake_file.close()



gan_dir = '/data/data1/users/fkonid/data_gan_adnorfirst/gan/'

get_perms(gan_dir, 'nor')
get_perms(gan_dir, 'ad')
