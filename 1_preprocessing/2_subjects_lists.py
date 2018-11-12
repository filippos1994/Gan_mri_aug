# Stores the names of subjects to files.

from os import listdir

type_dict = {'ad', 'nor'}
set_dict = {'train', 'test', 'valid'}
list_dir = "subjects_list/"

for dtype in type_dict:
	for dset in set_dict:

		ttv_split_dir = "ttv_split_{}/{}/{}/".format(dtype, dset, dtype)
		name_list = listdir(ttv_split_dir)		
		subjects_path = list_dir + 'subjects_{}_{}.txt'.format(dset, dtype)

		with open(subjects_path, 'w') as subjects_file:
			
			for name in name_list:
				subjects_file.write(name + "\n")
