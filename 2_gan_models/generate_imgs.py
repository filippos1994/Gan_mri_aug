# Given a .h5 model (not just weights!) as input, produces NUM generated images
import argparse
import os
import numpy as np
import PIL.Image as Image
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial


NUM = 60000 # how many images we want to generate

TYPE = 'ad'
BATCH_SIZE = 32 # irrelevant for the actual computations, just for naming conventions to access the model
LATENT_SIZE = 128 # latent size used during training, fixed
EPOCHS = 400
MODEL = 'generator_{}_{}_{}_{}.h5'.format(TYPE, BATCH_SIZE, LATENT_SIZE, EPOCHS)
IMG_DIR = '{}/'.format(TYPE) # change at will

generator = load_model(MODEL)

noise = np.random.rand(NUM, LATENT_SIZE)
gen_imgs = generator.predict(noise)
gen_imgs = (gen_imgs * 127.5) + 127.5 # rescale

lst = []
for i in range(NUM):
	lst.append(i)

images = gen_imgs[:,:,:,0]

for img, i in zip(images, lst):
	#print(img.shape, i)
	im = Image.fromarray(img)
	if (im.mode != 'L'):
		im = im.convert('L')
	im.save(IMG_DIR + 'fake_{}_{}.png'.format(TYPE,i))
