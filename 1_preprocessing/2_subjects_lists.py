from os import listdir

type_dict = {'ad', 'nor'}
set_dict = {'train', 'test', 'valid'}
list_dir = "subjects_list/"

for dtype in type_dict:
	for dset in set_dict:

		ttv_split_dir = "/home/filippos/diplomatiki/adni/nifti/all/ttv_split_{}/{}/{}/".format(dtype, dset, dtype)
		name_list = listdir(ttv_split_dir)		
		subjects_path = list_dir + 'subjects_{}_{}.txt'.format(dset, dtype)

		try:
			subjects_file = open(subjects_path, 'w')
		except IOError:
			# If not exists, create the file
			subjects_file = open(subjects_path, 'w+')

		for name in name_list:
			subjects_file.write(name + "\n")

		subjects_file.close()




#ttv_split_dir = "/home/filippos/diplomatiki/adni/nifti/all/ttv_split_{}/{}/".format(dtype, dset)

"""name_list = []
for _, _, files in walk(ttv_split_dir):
	for file in files:
		name_list.append(file.split('/')[0])

name_list = list(set(name_list))
print(name_list)
"""

#name_list = listdir(ttv_split_dir)
#print(name_list)




#visits_list = ["s1/e1", "s1/e2", "s2/e1", "s2/e2", "s2/e3"]
#name_list = []

"""for visit in visits_list:
	name_list.append(visit.split('/')[0])

print(name_list)

name_list = list(set(name_list))
print(name_list)
"""