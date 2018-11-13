# Randomly splits subjects on training, validation and test sets. 

import numpy as np

from os import listdir, path
from shutil import copytree


nii_dir = "nor" # original directories were "ad" (AD patient) and "nor" (Normal)
split_dir = "ttv_split_{}".format(nii_dir)

train_dir = path.join(split_dir, "train")
test_dir  = path.join(split_dir, "test")
valid_dir = path.join(split_dir, "valid")

patients = listdir(nii_dir)

for patient in patients:
	if nii_dir == "nor":
		keep_patient = np.random.rand() <= 0.5804 # to balance Normal subjects with AD patients
	else:
		keep_patient = True

	if keep_patient:
		split = np.random.rand()

		if split <= 0.8333: # ~25.000/30.000 images to train set
			current_dir = train_dir
			src = path.join(nii_dir, patient)
			dst = path.join(current_dir, path.join(nii_dir, patient))

			copytree(src,dst)
#			print("train")
		elif split <=0.9333: # ~3.000/30.000 images to valid set
			current_dir = valid_dir
			src = path.join(nii_dir, patient)
			dst = path.join(current_dir, path.join(nii_dir, patient))

			copytree(src,dst)
#			print("valid")
		else: # ~2.000/30.000 images to test set
			current_dir = test_dir
			src = path.join(nii_dir, patient)
			dst = path.join(current_dir, path.join(nii_dir, patient))

			copytree(src,dst)
#			print("test")
