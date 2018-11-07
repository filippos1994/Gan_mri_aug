import shutil
from os import walk, path

dtype = "nor" # dtype is either "ad" or "nor"
png_dir = "ttv_split_{}_png/".format(dtype)
seq_dir = "ttv_split_{}_png_seq/".format(dtype)
for root, _, files in walk(png_dir):
#	[print(file) if (str(file).endswith("{}.png".format(str(i))) for i in range(50,150)) for file in files]
	for file in files:
		if "test" in root:
			split = "test/{}/".format(dtype)
		elif "train" in root:
			split = "train/{}/".format(dtype)
		elif "valid" in root:
			split = "valid/{}/".format(dtype)

		file_lst = file.split("_") # get {number}.png
		sequence = file_lst[-1]
		sequence = sequence.split('.')[0] # get {number}

		if len(files) == 192:
			if int(sequence) >= 95 and int(sequence) <= 140:
				shutil.copy(path.join(root, file), path.join(seq_dir, split))
		elif len(files) == 240:
			if int(sequence) >= 100 and int(sequence) <= 180:
				shutil.copy(path.join(root, file), path.join(seq_dir, split))
		elif len(files) == 256:
			if int(sequence) >= 120 and int(sequence) <= 200:
				shutil.copy(path.join(root, file), path.join(seq_dir, split))
