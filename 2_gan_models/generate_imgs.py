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












"""r, c = 5, 5 # an to allaksw na allaksw kai ta katw!!!
fixed_noise = np.random.rand(r * c, 100)
np.save(output_dir + "gan_nor_fixed_array.npy", fixed_noise)
def generate_images(generator_model, output_dir, epoch, fixed = False):
    r, c = 5, 5
    if not fixed:
        noise = np.random.rand(r * c, 100)
        output_dir += "rand/"
    else:
        noise = fixed_noise
        output_dir += "fixed/"
    gen_imgs = generator_model.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = (gen_imgs * 127.5) + 127.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    fig.savefig(outfile)
    plt.close()"""

