# Resize all images underneath directory 'data' to (160, 192), through Lanczos resampling.

import numpy as np

from os import path, walk
from PIL import Image

for root, _, files in walk("data"):
	for file in files:
		img = Image.open(path.join(root, file))
		img = img.resize((160,192), resample = Image.LANCZOS)

		dst = root.replace("data", "data_resized")
		img.save(path.join(dst, file))
