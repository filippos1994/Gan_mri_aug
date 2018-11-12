# main classification script

import keras
#import matplotlib.image as mpimg
import numpy as np
import itertools
import tensorflow as tf

from gan_dir_it import *

from imgaug import augmenters as iaa

from custom_resnet import ResNet18
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image

from math import ceil
from my_opts import opt
from os import listdir, makedirs, path, walk


if opt.mem_perc < 1.0:
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = opt.mem_perc
  K.tensorflow_backend.set_session(tf.Session(config = config))


################################

# PART ONE: set names, directories etc

################################ 

data_dir = opt.data_dir
target_size = opt.target_size
stats_dir = data_dir + 'stats_train/' 


name = 'bs_{}'.format(opt.batch_size)

dropout    = opt.dropout_prob > 0.0
fully_conn = opt.neurons > 0
augment    = opt.aug_prob > 0.0
fake_ratio = opt.fake_ratio

if (fake_ratio <= 0.0) or (fake_ratio > 2.0):
  print('Wrong fake ratio given. Should be greater than 0.0 and no greater than 2.0. Exiting.')
  import sys
  sys.exit()

if fake_ratio < 1.0:
  fake_ratio_str = '0' + str(int(opt.fake_ratio*100))
else:
  fake_ratio_str = str(int(opt.fake_ratio*100))

stats_path = stats_dir + 'stats_{}.txt'.format(fake_ratio_str)


if dropout:
  name += '_dr_{}'.format(int(opt.dropout_prob * 100))
if fully_conn:
  name += '_n_{}'.format(opt.neurons)

arch_name = name

name += '/{}'.format(fake_ratio_str)

if augment:
  name = 'aug_no_cont/aug_prob_{}/'.format(int(opt.aug_prob * 100)) + name
else:
  name = 'no_aug/' + name
name += '/'

makedirs(name)
print(name)


if opt.weights_dir is None:
  weights_path = name
  print('yes')
else:
  print('no')
  weights_path = name + opt.weights_dir
  if not opt.weights_dir.endswith('/'):
    weights_path += '/'
  makedirs(weights_path)

best_weights_path = weights_path + 'best_weights_{}.h5'.format(arch_name)
print(best_weights_path)

if opt.log_dir is not None:
  log_dir = name + opt.log_dir
  makedirs(log_dir)

batch_size = opt.batch_size

################################

# PART TWO: preprocess input data, set pipelines etc

################################ 


def data_mean_std():
    names = []
    for root, dirs, files in walk(data_dir):
        if "train" in root:
            for file in files:
                names.append(root + '/' + file)
                
    array = []
    for name in names:
        img = image.load_img(name, grayscale = True, target_size = target_size)

        img_array = np.asarray(img)
        array.append(img_array)

    array = np.asarray(array)
    return array.mean(), array.std()


aug_prob = opt.aug_prob
seq = iaa.Sequential([
        iaa.Fliplr(aug_prob),
	iaa.Sometimes(aug_prob, iaa.Multiply((0.9, 1.1))),
        iaa.Sometimes(aug_prob, iaa.Affine(
                  scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # scale images to 80-120% of their size
                  translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # translate by -5 to +5 percent (per axis)
                  rotate=(-5, 5),  # rotate by -5 to +5 degrees
                  cval=0,
                  mode='constant'
                  ))
      ])

#mean, std = data_mean_std()
#mean, std = 35.31741725575326, 44.92010876082367
stats_file = open(stats_path, 'r')
stats_lst = stats_file.read().split('\n')
mean = float(stats_lst[0])
std = float(stats_lst[1])

print(mean, std)
def normalize(x):
    return (x - mean) / std

def norm_and_aug(x):
    return seq.augment_image(normalize(x))


train_path = path.join(data_dir, "train_{}".format(fake_ratio_str))
valid_path = path.join(data_dir, "valid")

#gen = image.ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True)
if aug_prob > 0.0:
  train_gen = image.ImageDataGenerator(preprocessing_function = norm_and_aug)
#  train_gen = GANImageDataGenerator(preprocessing_function = norm_and_aug)
else:
  train_gen = image.ImageDataGenerator(preprocessing_function = normalize)
#  train_gen = GANImageDataGenerator(preprocessing_function = normalize)

valid_gen = image.ImageDataGenerator(preprocessing_function = normalize)


train_batches = train_gen.flow_from_directory(train_path, target_size = target_size, color_mode = 'grayscale', batch_size = batch_size)
valid_batches = valid_gen.flow_from_directory(valid_path, target_size = target_size, color_mode = 'grayscale', batch_size = batch_size)


################################

# PART THREE: load model

################################ 


tb = TensorBoard(log_dir = log_dir, batch_size = batch_size, write_graph = True)

lr_reducer = ReduceLROnPlateau(monitor='loss', factor=np.sqrt(0.1), patience=10, cooldown=0, min_lr=1e-5)

model_chkp = ModelCheckpoint(best_weights_path,
                            # monitor='val_quadratic_kappa_score',
                            monitor='val_acc',
                            save_best_only=True, save_weights_only=True, mode='max')


resnet = ResNet18()
#resnet = ResNet50()
#model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#model.summary()

#resnet.layers.pop()
#for layer in resnet.layers:
#    layer.trainable = False

x = resnet.layers[-1].output
if dropout:
  x = Dropout(opt.dropout_prob)(x)
if fully_conn:
  x = Dense(opt.neurons, activation = 'relu')(x)
  if dropout:
    x = Dropout(opt.dropout_prob)(x)

x = Dense(2, activation = 'softmax')(x)


ft_resnet = Model(resnet.input, x)
ft_resnet.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

print("model loaded")


################################

# PART FOUR: train model

################################ 


epochs = opt.epochs
train_samples = len(listdir(train_path + '/ad/')) + len(listdir(train_path + '/nor/'))
valid_samples = len(listdir(valid_path + '/ad/')) + len(listdir(valid_path + '/nor/'))

print(train_samples, valid_samples)

train_steps = ceil(train_samples/batch_size)
valid_steps = ceil(valid_samples/batch_size)

print (train_steps, valid_steps)

ft_resnet.fit_generator(train_batches, steps_per_epoch=train_steps,
                        epochs=epochs, callbacks = [tb, model_chkp, lr_reducer],
                        validation_data=valid_batches, validation_steps = valid_steps)

continue_model_path = weights_path + '{}_{}_eps.h5'.format(arch_name, epochs)
ft_resnet.save(continue_model_path)
