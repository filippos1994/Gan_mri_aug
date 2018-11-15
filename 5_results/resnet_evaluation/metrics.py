import matplotlib
matplotlib.use('Agg')

import keras
#import tensorflow
#import matplotlib.image as mpimg
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys
#import bcolz

from custom_resnet import ResNet18
from keras import regularizers
from keras.preprocessing import image
from keras.layers import Activation, BatchNormalization, Dense, Dropout, InputLayer
from keras.layers import LeakyReLU as lrelu
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from metrics_opts import opt
from sklearn.preprocessing import OneHotEncoder

from os import listdir,rename
from math import ceil

def calculate_metrics(data_dir = None, weights_path = None, arch_name = None, fake_ratio_str = None):

	if (data_dir is None) or (weights_path is None):
		print('Both parameters must be set. Exiting')
		sys.exit()

	if opt.fake_ratio == 0.0:
		mean, std = 35.31741725575326, 44.92010876082367
	else:
		stats_file = open(data_dir + 'stats_train/stats_{}.txt'.format(fake_ratio_str), 'r')
		stats_lst = stats_file.read().split('\n')
		mean = float(stats_lst[0])
		std = float(stats_lst[1])
	print(mean, std)
	def normalize(x):
		return (x - mean) / std

	batch_size = 32
	test_gen = image.ImageDataGenerator(preprocessing_function = normalize)
	test_batches = test_gen.flow_from_directory(data_dir + 'test/', target_size = (192,160),
							color_mode = 'grayscale', shuffle = False, batch_size = batch_size)

	#dir(test_batches)
	#test_batches.classes[200:300]

	res = ResNet18()
	#for layer in res.layers:
	#    layer.trainable = False
		
	x = res.layers[-1].output

	##### kathe fora tha thelei allagi, analoga me tin arxitektonikh, alliws tha vgazei error
	if opt.dropout_prob > 0.0:
		x = Dropout(opt.dropout_prob)(x)
	if opt.neurons > 0:
		x = Dense(opt.neurons, activation = 'relu')(x)
		if opt.dropout_prob > 0.0:
			x = Dropout(opt.dropout_prob)(x)

	x = Dense(2, activation = 'softmax')(x)

	model = Model(res.input, x)
	model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

	model.load_weights(weights_path)

	probs = model.predict_generator(test_batches, verbose=1)

	#print(probs.shape)
	#print(type(probs))

	expected_labels = test_batches.classes

	our_predictions = probs[:,0]
	our_labels = np.round(1-our_predictions)

	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(expected_labels, our_labels)

	from sklearn.metrics import classification_report, accuracy_score
	print(classification_report(expected_labels, our_labels, digits=4))
	acc_score = accuracy_score(expected_labels, our_labels)
	print(acc_score)

	s = classification_report(expected_labels, our_labels, digits=4)
	print(s)

	f = open('classification_report_{}.txt'.format(arch_name), 'w+')
	f.write(s)
	f.close()

	splt = s.split('\n')
	len(splt)
	#print(splt[2])
	stats = splt[-2].replace('    ', ' ')
	stats = '{0:.4f} '.format(acc_score) + stats.split('  ')[1]
	print(stats)

	f = open('metrics_{}.txt'.format(arch_name), 'w+')
	f.write(stats)
	f.close()
	#print(stats[1:-2])

	def plot_confusion_matrix(cm, arch_name, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		(This function is copied from the scikit docs.)
		"""
		plt.figure()
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print(cm)
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

		#plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig('cm_{}.png'.format(arch_name))

	plot_confusion_matrix(cm, arch_name, test_batches.class_indices)


## main
if opt.fake_ratio == 0.0:
	weights_path = '../final_no_gan/'
	data_dir = '/data/data1/users/fkonid/data_resized/'
	fake_ratio_str = None
else:
	weights_path = '../final_with_gan/'
	data_dir = '/data/data1/users/fkonid/data_gan_adnorfirst/'
	if opt.fake_ratio < 1.0:
		fake_ratio_str = '0{}'.format(int(opt.fake_ratio*100))
	else:
		fake_ratio_str = str(int(opt.fake_ratio*100))

if opt.aug_prob == 0.0:
	weights_path += 'no_aug/'
else:
	weights_path += 'aug_no_cont/aug_prob_{}/'.format(int(opt.aug_prob*100))


arch_name = 'bs_{}'.format(opt.batch_size)

if opt.dropout_prob > 0.0:
	arch_name += '_dr_{}'.format(int(opt.dropout_prob*100))
if opt.neurons > 0:
	arch_name += '_n_{}'.format(opt.neurons)

weights_path += arch_name + '/'

if opt.aug_prob > 0.0:
	arch_name = 'aug_prob_{}_'.format(int(opt.aug_prob*100)) + arch_name

if fake_ratio_str is not None:
	weights_path += fake_ratio_str + '/'
	weights_path += 'best_weights_{}.h5'.format(arch_name)
	arch_name = 'fake_{}_'.format(fake_ratio_str) + arch_name
else:
	weights_path += 'best_weights_{}.h5'.format(arch_name)

print(weights_path)
print(data_dir)
print(arch_name)

calculate_metrics(data_dir, weights_path, arch_name, fake_ratio_str)
